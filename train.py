import argparse
import torch
from typing import Any, Optional
from ultralytics import YOLO

class YOLOModelHandler:
    def __init__(self, model_name: str):
        self.model_name: str = model_name
        self.model: Optional[YOLO] = None
        self.load_model()

    def load_model(self) -> None:
        if self.model_name.endswith('.yaml'):
            # Build a new model from scratch
            self.model = YOLO(self.model_name)
        elif self.model_name.endswith('.pt'):
            # Load a pre-trained model (recommended for training)
            self.model = YOLO(self.model_name)

            # # Check and set the device
            # if torch.backends.mps.is_available():
            #     self.device = torch.device("mps")  # Use MPS if available
            # elif torch.cuda.is_available():
            #     self.device = torch.device("cuda")  # Use CUDA if MPS is unavailable but CUDA is
            # else:
            self.device = torch.device("cpu")   # Use CPU if neither MPS nor CUDA is available

        # Load the model onto the specified device
        if self.model:
            self.model.to(self.device)
        else:
            raise ValueError("Unsupported model format. Use '.yaml' or '.pt'")

    def train_model(self, data_config: str, epochs: int) -> None:
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Train the model
        self.model.train(data=data_config, epochs=epochs)

    def validate_model(self) -> Any:
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Evaluate model performance on the validation set
        return self.model.val()

    def predict_image(self, image_path: str) -> Any:
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Predict on an image
        return self.model(image_path)

    def export_model(self, export_format: str = "onnx") -> str:
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Export the model to the desired format
        return self.model.export(format=export_format)
    
    def save_model(self, save_path: str) -> None:
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Save the model to the specified path
        torch.save(self.model.state_dict(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handle YOLO model training, validation, prediction, and exporting.')
    
    parser.add_argument('--data_config', type=str, default='data.yaml', help='Path to the data configuration file')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--model_name', type=str, default='yolov8n.pt', help='Name of the YOLO model file')
    parser.add_argument('--export_format', type=str, default='onnx', help='Format to export the model to')
    parser.add_argument('--onnx_path', type=str, default=None, help='Path to save the exported ONNX model')
    parser.add_argument('--pt_path', type=str, default='models/model.pt', help='Path to save the trained model in .pt format')

    args = parser.parse_args()

    handler = YOLOModelHandler(args.model_name)

    try:
        handler.train_model(data_config=args.data_config, epochs=args.epochs)
        metrics = handler.validate_model()
        results = handler.predict_image("")
        export_path = handler.export_model(export_format=args.export_format) if args.onnx_path is None else args.onnx_path
        handler.save_model(args.pt_path)
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    print("Prediction results:", results)
    print(f"{args.export_format.upper()} model exported to:", export_path)
    print(f"Model saved to: {args.pt_path}")