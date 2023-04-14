import os
import time

class Logger:
    def __init__(self, log_path):
        # 创建文件夹，防止文件夹不存在
        folder, file = os.path.split(log_path)
        os.makedirs(folder, exist_ok=True)

        self.log_file = log_path
        self.log = open(log_path, 'a')
        now = time.strftime('%Y%m%d-%H%M%S', time.localtime(time.time()))
        self.log.write("{}  Logger Init.".format(now))

    def __call__(self, *content):
        try:
            content = list(content)
            for i in range(len(content)):
                content[i] = str(content[i])
            content = ' '.join(content)
        except Exception as e:
            content = "Converting content to String encountered an Error:\n"+str(e)
        self.log.write(content + '\n')
        print(content)

    def __del__(self):
        self.log.close()