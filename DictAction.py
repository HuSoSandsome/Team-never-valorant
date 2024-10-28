import argparse

class DictAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        # 初始化一个字典存储键值对
        result_dict = {}
        for kv in values:
            # 分割键和值
            key, value = kv.split('=')
            result_dict[key] = value
        # 将解析后的字典存入namespace
        setattr(namespace, self.dest, result_dict)

# 示例代码，展示如何使用 DictAction
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DictAction Example")
    parser.add_argument('--config', nargs='+', action=DictAction, help="Key-value pairs")

    args = parser.parse_args()
    print(args.config)
