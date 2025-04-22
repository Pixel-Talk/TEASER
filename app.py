import os
import time
import gradio as gr
from pathlib import Path
from functools import partial
from utils.rprint import rlog as log

OUTPUT_DIR = 'results'
DEVICES = '0'
IN_IMAGE_DIR = 'assets/demo/images'
OUTPUT_DIR = 'results/tracking_results'
EXCHANGE_DIR = 'results/exchanging_results'


def run_cmd(cmd, current_dir):
    log(cmd)
    this_day = time.strftime('%Y-%m-%d')
    cur_log_fp = os.path.join(current_dir, 'logs', f'log-{this_day}.log')
    if not os.path.exists(cur_log_fp):
        os.makedirs(os.path.dirname(cur_log_fp), exist_ok=True)
    with open(cur_log_fp, 'a') as f:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{current_time}]' + cmd + '\n')
    os.system(cmd)


def check_process_status(source_image):
    """Check if the processing is complete and return the result video if available"""
    if not OUTPUT_DIR or not os.path.exists(OUTPUT_DIR):
        return "Processing hasn't started yet.", None
    
    src_name = os.path.splitext(os.path.basename(source_image))[0]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, OUTPUT_DIR)
    output_case_dir = os.path.join(output_dir, src_name)
    out_img_fp = os.path.join(output_case_dir, 'reconstruct.jpg')
    out_obj_fp = os.path.join(output_case_dir, 'head.obj')
        
    log('Try to find => ' + output_case_dir)

    if not os.path.exists(out_img_fp) and not os.path.exists(out_obj_fp):
        return "Still processing... You can leave but keep this page open. ⏳", None, None
        
    return "Processing completed successfully! 🎉", out_img_fp, out_obj_fp


def process_inference(source_image, progress=gr.Progress()):
    
    progress(0.1, desc="Files saved, starting processing...")
    
    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, OUTPUT_DIR)

        my_run_cmd = partial(run_cmd, current_dir=current_dir)

        src_name = os.path.splitext(os.path.basename(source_image))[0]
        output_case_dir = os.path.join(output_dir, src_name)
        os.makedirs(output_case_dir, exist_ok=True)
        out_img_fp = os.path.join(output_case_dir, 'reconstruct.jpg')
        out_obj_fp = os.path.join(output_case_dir, 'head.obj')
        
        if os.path.exists(out_img_fp) and os.path.exists(out_obj_fp):
            log(f'🐶 Result has been generated, skipping...')
        else:
            src_img_fp = os.path.join(current_dir, source_image)
            my_run_cmd(f'PYTHONPATH=.  python cli.py -i {src_img_fp} ' + 
                                            f' -o {output_case_dir}')
            log(f'Done! The result is saved in {output_case_dir}')

        progress(1.0, desc="🎉 Done! ")
        
        return "🎉 Done! ", out_img_fp, out_obj_fp
                
    except Exception as e:
        return f"Error occurred: {str(e)}", None, None


def process_image_exp(source_image, driven_image, progress=gr.Progress()):
    log("Processing image...")
    progress(0.1, desc="Processing image...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, EXCHANGE_DIR)

    my_run_cmd = partial(run_cmd, current_dir=current_dir)

    src_name = os.path.splitext(os.path.basename(source_image))[0]
    drv_name = os.path.splitext(os.path.basename(driven_image))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_case_fp = os.path.join(output_dir, f'{src_name}__{drv_name}-exp.jpg')

    if os.path.exists(output_case_fp):
        log(f'Found existing result: {output_case_fp}')
        return  "🎉 Done! ", output_case_fp

    src_img_fp = os.path.join(current_dir, source_image)
    drv_img_fp = os.path.join(current_dir, driven_image)
    my_run_cmd(f'PYTHONPATH=.  python cli.py -s {src_img_fp} ' + 
                                          f' -d {drv_img_fp} ' + 
                                          f' -xo {output_case_fp} --type exp') 
    progress(1.0, desc="🎉 Done! ")

    return None, output_case_fp


def process_image_token(source_image, driven_image, progress=gr.Progress()):
    log("Processing image...")
    progress(0.1, desc="Processing image...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, EXCHANGE_DIR)

    my_run_cmd = partial(run_cmd, current_dir=current_dir)

    src_name = os.path.splitext(os.path.basename(source_image))[0]
    drv_name = os.path.splitext(os.path.basename(driven_image))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_case_fp = os.path.join(output_dir, f'{src_name}__{drv_name}-token.jpg')

    if os.path.exists(output_case_fp):
        log(f'Found existing result: {output_case_fp}')
        return  "🎉 Done! ", output_case_fp

    src_img_fp = os.path.join(current_dir, source_image)
    drv_img_fp = os.path.join(current_dir, driven_image)
    my_run_cmd(f'PYTHONPATH=.  python cli.py -s {src_img_fp} ' + 
                                          f' -d {drv_img_fp} ' + 
                                          f' -xo {output_case_fp} --type token') 
    progress(1.0, desc="🎉 Done! ")

    return None, output_case_fp


# Create the Gradio interface
with gr.Blocks(title="TEASER demo", css="""
    .image-container { 
        position: relative; 
        display: inline-block;
        width: 100%;
        height: 100%;
    }
    .overlay-button { 
        position: absolute !important; 
        top: 50% !important; 
        left: 50% !important; 
        transform: translate(-50%, -50%) !important;
        opacity: 0;
        transition: opacity 0.3s;
        background: rgba(0,0,0,0.7) !important;
        color: white !important;
        border: none !important;
        z-index: 1;
    }
    .image-container:hover .overlay-button { 
        opacity: 1; 
    }
    .gradio-image {
        position: relative !important;
        width: 100% !important;
        height: 100% !important;
    }
    .scrollable-column {
        height: 300px !important;
        overflow-y: auto !important;
        padding-right: 10px;
    }
    .scrollable-column::-webkit-scrollbar {
        width: 8px;
    }
    .scrollable-column::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    .scrollable-column::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    .scrollable-column::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    .circular-image {
        border-radius: 50% !important;
        overflow: hidden !important;
        width: 200px !important;
        height: 200px !important;
        object-fit: cover !important;
    }
""") as demo:
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Image("samples/teaser.png", show_label=False, height=170, container=False, interactive=False, elem_classes=["circular-image"])
        with gr.Column(scale=6):
            gr.Markdown("""
            ## TEASER: Token-EnhAanced Spatial Modeling for Expression Reconstruction
            """)
            with gr.Row():
                gr.Markdown("""""")
            gr.Markdown("""
                        <div style="text-align: center;">
                        <div style="display: inline-block;">
                        <a href='https://arxiv.org/abs/2502.10982' style='padding-left: 0.5rem;'>
                            <img src='https://img.shields.io/badge/arXiv-2502.10982-brightgreen' alt='arXiv'></a></div><div style="display: inline-block;">
                        <a href='https://julia-cherry.github.io/TEASER-PAGE/' style='padding-left: 0.5rem;'>
                            <img src='https://img.shields.io/badge/Website-Project Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'></a></div></div>

                        > TEASER reconstructs precise 3D facial expression and generates high-fidelity face image through estimating hybrid parameters for 3D facial reconstruction.
                        
                        **Upload a source image we will generate its 3D expressions, we also provide some applications.**
                        """)
    
    with gr.Tab("3D Expression Reconstruction"):
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(label="Input Image", type="filepath", height=600)
                process_btn = gr.Button("Inference", variant="primary")
            
            with gr.Column():
                check_btn = gr.Button("Check Progress 🔄", variant="secondary")
                output_message = gr.Textbox(label="Status")
                output_image = gr.Image(label="Generated results (Rendered Mesh | Generated Image)")
                model_rlt = gr.Model3D(label="Generated 3D expression model")
                
                with gr.Row():
                    process_btn.click(
                            fn=process_inference,
                            inputs=[source_image],
                            outputs=[output_message, output_image, model_rlt]
                        )
                    
                    check_btn.click(
                        fn=check_process_status,
                        inputs=[source_image],
                        outputs=[output_message, output_image, model_rlt]
                    )
        
        gr.Markdown("---")
        gr.Markdown("### Example Input Images (Click to use)")
        with gr.Row(elem_classes=["scrollable-column"]):
            example_images = [f for f in sorted(os.listdir(IN_IMAGE_DIR)) if f.endswith(('.png', '.jpg', '.jpeg'))]
            for img in example_images:
                img_path = os.path.join(IN_IMAGE_DIR, img)
                with gr.Column(elem_classes=["image-container"]):
                    gr.Image(value=img_path, show_label=True, label=img, height=250)
                    select_img_btn = gr.Button("Use this image", size="sm", elem_classes=["overlay-button"])
                    select_img_btn.click(
                        fn=lambda x=img_path: x,
                        outputs=[source_image]
                            )

    with gr.Tab("Exchange Token and Expression"):
        gr.Markdown("### Input Image")
        with gr.Row():
            with gr.Column():
                source_image_1 = gr.Image(label="Source Image", type="filepath", height=300)
                driven_image = gr.Image(label="Driven Image", type="filepath", height=300)
                with gr.Row():
                    process_x_exp_btn = gr.Button("Exchange Expression", variant="primary")
                    process_x_token_btn = gr.Button("Exchange Token", variant="primary")
            with gr.Column():
                output_message_1 = gr.Textbox(label="Status", interactive=False)
                exchange_result = gr.Image(label="Generated results (Rendered Mesh | Generated Image)")

        process_x_exp_btn.click(fn=process_image_exp, inputs=[source_image_1, driven_image], outputs=[output_message_1, exchange_result])
        process_x_token_btn.click(fn=process_image_token, inputs=[source_image_1, driven_image], outputs=[output_message_1, exchange_result])

        with gr.Row():
            with gr.Column():
                gr.Markdown("---")
                gr.Markdown("### Example Source Images (Click to use)")
                with gr.Row(elem_classes=["scrollable-column"]):
                    example_images = [f for f in sorted(os.listdir(IN_IMAGE_DIR)) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    for img in example_images:
                        img_path = os.path.join(IN_IMAGE_DIR, img)
                        with gr.Column(elem_classes=["image-container"]):
                            gr.Image(value=img_path, show_label=True, label=img, height=250)
                            select_img_btn = gr.Button("Use this image", size="sm", elem_classes=["overlay-button"])
                            select_img_btn.click(
                                fn=lambda x=img_path: x,
                                outputs=[source_image_1]
                                    )
            with gr.Column():
                gr.Markdown("---")
                gr.Markdown("### Example Driven Images (Click to use)")
                with gr.Row(elem_classes=["scrollable-column"]):
                    example_images = [f for f in sorted(os.listdir(IN_IMAGE_DIR)) if f.endswith(('.png', '.jpg', '.jpeg'))]
                    for img in example_images:
                        img_path = os.path.join(IN_IMAGE_DIR, img)
                        with gr.Column(elem_classes=["image-container"]):
                            gr.Image(value=img_path, show_label=True, label=img, height=250)
                            select_img_btn = gr.Button("Use this image", size="sm", elem_classes=["overlay-button"])
                            select_img_btn.click(
                                fn=lambda x=img_path: x,
                                outputs=[driven_image]
                                    )

    gr.Markdown("""
        --- 
        📝 **Citation**
        <br>
        If our work is useful for your research, please consider citing:
        ```bibtex
        @inproceedings{liu2025TEASER,
            title={TEASER: Token Enhanced Spatial Modeling for Expressions Reconstruction},
            author={Liu, Yunfei and Zhu, Lei and Lin, Lijian and Zhu, Ye and Zhang, Ailing and Li, Yu},
            booktitle={ICLR},
            year={2025}
            }
        ```
        📧 **Contact**
        <br>
        If you have any questions, please feel free to send a message to <b>liuyunfei.cs@gmail.com</b> or open an issue on the [Github repo](https://github.com/Pixel-Talk/TEASER).
            """)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Whether to share the app.")
    args = parser.parse_args()

    demo.launch(allowed_paths=["."], share=args.share) 