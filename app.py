# app.py — Nova Harness 桌面入口
# 用法：python app.py
# 若安装了 pywebview 则弹出原生桌面窗口，否则自动打开默认浏览器。
import os
import socket
import threading
import time
import webbrowser

# 固定工作目录到项目根（config/checkpoints 都是相对路径）
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from harness.server import serve  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    httpd = serve(port=port)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    try:
        import webview  # pywebview，可选
        webview.create_window("Nova Harness", url, width=1180, height=780)
        print("[app] 桌面窗口已打开，关闭窗口即退出。")
        webview.start()
    except ImportError:
        webbrowser.open(url)
        print("[app] 已在浏览器打开 " + url)
        print("[app] 想要原生桌面窗口可执行: pip install pywebview")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("[app] bye")


if __name__ == "__main__":
    main()
