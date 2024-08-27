import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC1 = "test/tuiasi/robot/feedback"
TOPIC2 = "test/tuiasi/robot/DoTaskCmd"
client_is_connected = False


def on_connect(client, userdata, flags, rc):
    global client_is_connected
    print("Connected to the Broker")
    client.subscribe(TOPIC1)
    client_is_connected = True


def on_message(client, userdata, msg):
    print(f"Received msg: {msg.payload.decode()} from {msg.topic}")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)

client.loop_start()


def calculeaza_coordonate(pozitii: str):
    litere = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1, 'h': 0}
    cifre = {'8': 0, '7': 1, '6': 2, '5': 3, '4': 4, '3': 5, '2': 6, '1': 7}

    if(pozitii == "turn"):
        coord1_x = -1.8
        coord1_y = 1.8
        coord2_x = -1.9
        coord2_y = 3.8
    else:

        prima_pozitie = pozitii[:2]
        a_doua_pozitie = pozitii[2:]


        coord1_x = litere[prima_pozitie[0].lower()]
        coord1_y = cifre[prima_pozitie[1]]

        if a_doua_pozitie == "outside":
            coord2_x = 9
            coord2_y = 3
        else:
            coord2_x = litere[a_doua_pozitie[0].lower()]
            coord2_y = cifre[a_doua_pozitie[1]]

    rezultat = f"{coord1_x} {coord1_y} {coord2_x} {coord2_y}"
    return rezultat


count = 0
while True:
    # mesaj = input("Publish: ")
    # client.publish(TOPIC2, mesaj)
    if count < 1 and client_is_connected:
        move = "e7e5"
        coords = calculeaza_coordonate(move)
        print(f"Coordinates: {coords}")
        client.publish(TOPIC2, coords)
        count = count + 1

# client.loop_forever()
