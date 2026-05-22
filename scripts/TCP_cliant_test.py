import socket
 
HOST = "127.0.0.1"
PORT = 29999
 
s = socket.socket()
s.connect((HOST, PORT))
 
while True:
    input("Enter → 캡처 ")
    s.sendall(b"C\n")
    print("수신:", s.recv(4096).decode().strip())
 
s.close()
 