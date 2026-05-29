# TODO


模型应该完全从机械臂本体感知获得 gripper width，然后用 sigma 读取的 action 作为输出。

这个应该需要修改：`record`、`dataset`、`agent`、`eval`。

- git commit 规范

- 修改record：joint action? gripper
- 修改eval：支持异步推理?RTC?
- 修改config：使用state_config和action_config支持修改state和输出的类型。之后修改eval
- 思考一下怎么修改训练集大小
- profile：查看显存占用，看看能否修改batchsize让训练更快
- 尝试支持dp
- record的时候大家好像都是获得上一帧，可能可以改改？

- 支持更加丰富的训练参数
    - 使用joint/tcppose(quat)/tcppose(eular)，是否使用delta，以及delta的具体算法
    - 输出action / next tcppose / next joint，是否使用delta
    - 如何指定训练集，如何限制训练集的大小
    - 如何给图像做增强
    - 寻找需要多少数据，用什么配置最好
    - 调试显卡占用和训练速度

- compute_norm_stats
    - 保存的时候支持修改repo_id
    - 保存的路径添加一些state参数
    - 固定显卡=1？
- 训练时
    - 传入episode_index
    - 读取对应的norm stats
    - profile
- 现在这个config有点垃圾。他支持靠default生成配置。对于做对照实验非常麻烦，我应该自己写一个general的config，然后从我的config推到出openpi的config。我自己的config应该保存在ckpt旁边，这样eval的时候也比较方便。这个东西应该叫config builder？搞一个类