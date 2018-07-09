from socket import *

servername = '35.196.250.59'
serverport = 12000
clientSocket = socket(AF_INET, SOCK_DGRAM)
message = input('Type in lowercase sentence: ')
clientSocket.sendto(message.encode(), (servername, serverport))
print("Waiting for response from server.")
modifiedMessage, serverAddress = clientSocket.recvfrom(2048)
print(modifiedMessage)
clientSocket.close()
