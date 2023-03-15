import http.server
import socketserver
from secrets import SystemRandom

handler = http.server.SimpleHTTPRequestHandler

def _random(n, m):
  return SystemRandom().randint(n, m)

a = _random(0, 8080)

with socketserver.TCPServer(('', a), handler) as httpd:
  print(f'Server listening on port {a}...')
  httpd.serve_forever()
