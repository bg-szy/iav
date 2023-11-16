import os

import torch
import torch.onnx

from models import TwinLite as Net


def export_pt(pth_file):
    model = Net.TwinLiteNet()
    loaded_model = torch.load(pth_file)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(loaded_model, strict=False)
    model.eval()
    example = torch.rand(1, 3, 360, 640).cuda()
    traced_script_module = torch.jit.trace(model.module, example)
    traced_script_module.save("twinlite.pt")


def export_onnx(pth_file, fp16=False):
    model = Net.TwinLiteNet()
    loaded_model = torch.load(pth_file)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(loaded_model, strict=False)
    model.eval()
    # data type nchw
    dummy_input1 = torch.randn(1, 3, 360, 640).cuda()
    input_names = ["image"]
    output_names = ["da", "ll"]

    torch.onnx.export(model.module, dummy_input1, "twinlite.onnx", input_names=input_names, output_names=output_names)
    if fp16:
        import onnx
        from onnxconverter_common import float16

        model = onnx.load("twinlite.onnx")
        model_fp16 = float16.convert_float_to_float16(model)
        onnx.save(model_fp16, "twinlite_fp16.onnx")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_file", type=str, default="twinlite.pth", help="pth file path")
    parser.add_argument("--target", type=str, default="onnx", help="include file type")
    parser.add_argument("--fp16", action="store_true", help="export fp16 onnx")
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.pth_file):
        print("pth file not exist")
        exit(0)
    if args.target == "pt":
        print("exporting pth to pt")
        export_pt(args.pth_file)
    elif args.target == "onnx":
        print("exporting pth to onnx")
        if args.fp16:
            print("exporting fp16 onnx")
            export_onnx(args.pth_file, fp16=True)
        else:
            export_onnx(args.pth_file)
    else:
        print("target file type error")
