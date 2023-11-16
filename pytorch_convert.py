import torch.onnx
from models import TwinLite as Net

if __name__ == "__main__":
    model = Net.TwinLiteNet()
    path = "twinlite.pth"
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(path))
    model.eval()
    dummy_input = torch.randn(1, 3, 360, 640, requires_grad=True).cuda()
    torch.onnx.export(model.module,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      "twinlite.onnx",  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['image'],  # the model's input names
                      output_names=['da', 'll'],  # the model's output names
                      )
    print(" ")
    print('Model has been converted to ONNX')