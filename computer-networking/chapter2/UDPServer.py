from socket import *

serverport = 12000
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('35.196.250.59', serverport))

print("The server is ready to reveive.")

while True:
    print("Waiting for new connections.")
    message, clientAddress = serverSocket.recvfrom(2048)
    print("Got message from {}".format(clientAddress))
    modifiedMessage = message.upper()
    serverSocket.sendto(modifiedMessage, clientAddress)
