# 网络和HTTP处理
import urllib.request
import urllib.parse
import http.server
import socketserver
import threading
import time

def urllib_demo():
    """网络请求示例"""
    print("=== 网络请求处理 ===")
    
    try:
        # 发送HTTP GET请求
        url = "https://httpbin.org/get"
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
            print(f"GET请求响应: {data[:200]}...")
    except Exception as e:
        print(f"网络请求失败: {e}")
    
    # URL编码
    params = {"name": "张三", "city": "北京"}
    encoded_params = urllib.parse.urlencode(params)
    print(f"URL编码参数: {encoded_params}")

def simple_http_server_demo():
    """简单HTTP服务器示例"""
    print("\n=== 简单HTTP服务器 ===")
    
    class MyHandler(http.server.SimpleHTTPRequestHandler):
        
        def do_GET(self):
            if self.path == '/hello':
                self.send_response(200)
                self.send_header('Content-type', 'text/html; charset=utf-8')
                self.end_headers()
                response = '<h1>你好，世界！</h1><p>这是一个简单的HTTP服务器</p>'
                self.wfile.write(response.encode('utf-8'))
            else:
                super().do_GET()
    
    def start_server():
        port = 8000
        with socketserver.TCPServer(("", port), RequestHandlerClass=MyHandler) as httpd:
            print(f"服务器启动在端口 {port}")
            print(f"访问 http://localhost:{port}/hello 查看示例页面")
            httpd.serve_forever()
    
    # 在后台线程启动服务器
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("HTTP服务器已在后台启动")
    time.sleep(1)  # 等待服务器启动

if __name__ == "__main__":
    urllib_demo()
    simple_http_server_demo()
    print("按 Ctrl+C 退出程序")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序退出")