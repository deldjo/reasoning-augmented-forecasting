from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        # 读取请求体
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        # 模拟返回真实的 LLM 结构（这里故意返回 -0.7 和 True）
        response = {
            "choices": [
                {
                    "message": {
                        "content": '{"sentiment": -0.7, "risk": true}'
                    }
                }
            ]
        }
        
        # 发送响应
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
        
    def log_message(self, format, *args):
        pass # 屏蔽控制台日志

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 8000), Handler)
    print("模拟 API 服务器已启动，监听端口 8000...")
    server.serve_forever()
