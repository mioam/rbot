```bash
# 使用相机拍摄rgbd图像，按q确认，保存在 outputs/scene0
uv run scripts/tools/capture.py
# 使用sam分割图像
uv run scripts/segment/file.py --path outputs/scene0/color.png
# 需要人工检查，删除不需要的物体
```

打开 sam3 object
```bash
# 生成mesh
python -m scripts.demo --path /ssd1/mzc/workspace/rbot/outputs/scene0
# 后处理，烘焙贴图
bash scripts/run.sh /ssd1/mzc/workspace/rbot/outputs/scene0
```