import argparse
import cv2
import matplotlib
import numpy as np
import os
import torch
import time

from depth_anything_v2.dpt import DepthAnythingV2

"""
-------------------------------------------------

            !!! 运行此代码 !!!
python run_webcam.py --save-video

-------------------------------------------------
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Webcam Demo')
    
    parser.add_argument('--input-size', type=int, default=256)
    parser.add_argument('--outdir', type=str, default='./results/webcam')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--camera-id', type=int, default=0, help='camera device ID')
    parser.add_argument('--save-video', action='store_true', help='save the processed video')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    print(f"Loading {args.encoder} model...")
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    print("Model loaded successfully")
    
    # Setup camera
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        exit(1)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = 15
    if frame_rate <= 0:  # 确保帧率有效
        frame_rate = 15  # 设置默认帧率
    
    # Setup video writer if needed
    out = None
    if args.save_video:
        os.makedirs(args.outdir, exist_ok=True)
        output_path = os.path.join(args.outdir, f"depth_webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width*2, frame_height))
        print(f"Saving video to {output_path} with {frame_rate} FPS")
    
    # Setup colormap
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    print("Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # 限制处理和保存的帧率，避免过快
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0 and args.save_video:
            actual_fps = frame_count / elapsed_time
            if actual_fps > frame_rate:
                # 如果实际帧率超过了设定帧率，暂停一下
                time.sleep(1/frame_rate)
        
        # Process the frame
        depth = depth_anything.infer_image(frame, args.input_size)
        
        # Normalize depth to 0-255 range
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Apply colormap if not grayscale
        if args.grayscale:
            depth_vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # Display original and depth side by side
        combined = np.hstack((frame, depth_vis))
        cv2.imshow('Depth Anything V2 - Webcam', combined)
        
        # Save video if requested
        if args.save_video and out is not None:
            out.write(combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    print("Exited successfully")