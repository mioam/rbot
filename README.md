# robot learning

## 环境

使用uv, 请参考 `pyproject.example.toml` 创建 `pyproject.toml`
注意：
- 这个文件固定了关键库（torch、numpy等）的版本，不要随意修改。
- 由于uv不支持不兼容的依赖，每次添加新的第三方库都需要精细修改。
- 使用uv sync更新环境，不要直接pip install
- `.envrc` 会添加必要的环境变量

## 目录

- docs: 文档
- playground: 被gitignore的目录，包括本地使用的临时脚本
- scripts: 可以直接执行的脚本，包括训练、测试等
- third_party: edit安装的包，可以直接修改以兼容环境，但是使用时推荐在src下写adapter。来源参考docs/git.md。
- src/rbot: 源代码

其他文件：
- ruff.toml: 使用ruff格式化代码
- checkpoints: 模型ckpt，用sync.sh和服务器同步
- .envrc: 环境变量
