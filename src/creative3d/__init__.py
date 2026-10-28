"""Creative3D v2 —— 参数化 3D 创意生成器。

模块划分：
    config      常量、主题、路径
    mesh        三角网格与 STL/OBJ 导出
    noise       值噪声 / fBm / 脊状噪声（numpy 向量化）
    generators  参数化形态生成器
    registry    形态登记表（单一数据源：界面 / 提示词 / 随机 / 导出共用）
    ideas       创意文案
    nlp         文本提示词解析
    imaging     图像分析（调色板 / 轮廓 / 高度场）
    render      视口光栅化（numpy 投影 + PIL 绘制）
    raster      高分辨率预览导出
    exporter    原子化导出与命名
    jobs        后台任务与进度
    settings    设置持久化
    session     单一状态源
"""

from .config import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
