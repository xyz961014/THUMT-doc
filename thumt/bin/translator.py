# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import six
import socket
import time
import torch
from tqdm import tqdm

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.utils as utils
from thumt.utils import update_cache


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, nargs="+",
                        help="Path of input file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, required=True, nargs="+",
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper parameters")
    parser.add_argument("--half", action="store_true",
                        help="Use half precision for decoding")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device_list=[0],
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        decode_batch_size=32,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters.lower())

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(args.vocabulary[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(args.vocabulary[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params


def convert_to_string(tensor, params):
    ids = tensor.tolist()

    output = []

    for wid in ids:
        if wid == 1:
            break
        output.append(params.mapping["target"][wid])

    output = b" ".join(output)

    return output


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def main(args):
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_params() for _ in range(len(model_cls_list))]
    params_list = [
        merge_params(params, model_cls.default_params())
        for params, model_cls in zip(params_list, model_cls_list)]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))]
    params_list = [
        override_params(params_list[i], args)
        for i in range(len(model_cls_list))]

    params = params_list[0]
    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    if args.half:
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Create model
    with torch.no_grad():
        model_list = []

        if len(args.input) == 1:
            mode = "infer"
            dataset = data.get_dataset(args.input[0], mode, params)
        else:
            # Teacher-forcing
            mode = "eval"
            dataset = data.get_dataset(args.input, mode, params)

        iterator = iter(dataset)
        idx = 0
        counter = 0
        pad_max = 1024
        top_beams = params.top_beams
        decode_batch_size = params.decode_batch_size

        # count eval dataset
        total_len = 0
        max_length = 0
        for sample in iterator:
            total_len += 1
            length = sample['source'].shape[1]
            if length > max_length:
                max_length = length
        iterator = iter(dataset)

        for param in params_list:
            if hasattr(param, "max_length"):
                param.max_length = min(param.max_length, max_length)
            else:
                param.max_length = max_length

        for i in range(len(args.models)):
            model = model_cls_list[i](params_list[i]).cuda()

            if args.half:
                model = model.half()

            model.eval()
            model.load_state_dict(
                torch.load(utils.latest_checkpoint(args.checkpoints[i]),
                           map_location="cpu")["model"])

            model_list.append(model)

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([decode_batch_size, top_beams, pad_max]).long()
                  for _ in range(dist.get_world_size())]

        if dist.get_rank() == 0:
            fd = open(args.output, "wb")
            pbar = tqdm(total=total_len)
            pbar.set_description("Translating to {}".format(args.output))
        else:
            fd = None

        states = [None for _ in model_list]
        if "cachedtransformer" in [model.name for model in model_list]:
            last_features = [None for _ in model_list]
        for model in model_list:
            if model.name == "cachedtransformer":
                model.encoder.cache.set_batch_size(params.decode_batch_size)
                model.decoder.cache.set_batch_size(params.decode_batch_size)

        while True:
            try:
                features = next(iterator)
                features = data.lookup(features, mode, params, from_torchtext=params.from_torchtext)

                if mode == "eval":
                    features = features[0]

                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float()
                }

                if mode == "eval":
                    features["target"] = torch.ones([1, 1]).long()
                    features["target_mask"] = torch.ones([1, 1]).float()

                batch_size = 0
            finally:
                for im, model in enumerate(model_list):
                    if model.name == "cachedtransformer":
                        features = update_cache(model, features, states[im], last_features[im], evaluate=True)
                        last_features[im] = features

            counter += 1

            # Decode
            if mode != "eval":
                seqs, _, states = utils.beam_search(model_list, features, params)
            else:
                seqs, _ = utils.argmax_decoding(model_list, features, params)

            # Padding
            pad_batch = decode_batch_size - seqs.shape[0]
            pad_beams = top_beams - seqs.shape[1]
            pad_length = pad_max - seqs.shape[2]
            seqs = torch.nn.functional.pad(
                seqs, (0, pad_length, 0, pad_beams, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(decode_batch_size):
                for j in range(dist.get_world_size()):
                    for k in range(top_beams):
                        n = size[j]
                        seq = convert_to_string(t_list[j][i][k], params)

                        if i >= n:
                            continue

                        if top_beams == 1:
                            fd.write(seq)
                            fd.write(b"\n")
                        else:
                            fd.write(str(idx).encode("utf-8"))
                            fd.write(b"\t")
                            fd.write(str(k).encode("utf-8"))
                            fd.write(b"\t")
                            fd.write(seq)
                            fd.write(b"\n")

                    idx = idx + 1

            if dist.get_rank() == 0:
                pbar.update(1)

        if dist.get_rank() == 0:
            pbar.close()
            fd.close()


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    world_size = infer_gpu_num(parsed_args.parameters)

    if world_size > 1:
        torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                    nprocs=world_size)
    else:
        process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
