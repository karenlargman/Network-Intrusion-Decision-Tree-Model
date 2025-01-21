import os
import subprocess
import urllib.request
import time
import tkinter as tk
from tkinter import scrolledtext
from scapy.all import sniff
import threading

packet_count_entry = None

def capture_packets(packet_count):
    def packet_handler(packet):
        packet_info = f"Timestamp: {packet.time}\n"
        packet_info += f"Length: {len(packet)}\n"
        packet_info += f"Protocol: {packet.proto}\n"
        if hasattr(packet, 'src'):
            packet_info += f"Source: {packet.src}\n"
        if hasattr(packet, 'dst'):
            packet_info += f"Destination: {packet.dst}\n"
        if hasattr(packet, 'sport'):
            packet_info += f"Source Port: {packet.sport}\n"
        if hasattr(packet, 'dport'):
            packet_info += f"Destination Port: {packet.dport}\n"
        if hasattr(packet, 'flags'):
            packet_info += f"TCP Flags: {packet.flags}\n"
        packet_info += "-" * 40 + "\n"
        
        text_box.insert(tk.END, packet_info)
        text_box.yview(tk.END)

    sniff(prn=packet_handler, count=packet_count, store=0)

def start_sniffing():

    packet_count = int(packet_count_entry.get())  # This line should now work
    sniff_thread = threading.Thread(target=capture_packets, args=(packet_count,))
    sniff_thread.start()

def main():
    global packet_count_entry  # Declare the global variable here
    
    print("Skipping Npcap installation (macOS compatible)...")
    print("Launching packet sniffer GUI...")

    # Set up Tkinter window
    root = tk.Tk()
    root.title("Network Packet Sniffer")
    
    packet_count_label = tk.Label(root, text="Number of packets to capture:")
    packet_count_label.pack(pady=10)

    packet_count_entry = tk.Entry(root)  # Now accessible in start_sniffing()
    packet_count_entry.pack(pady=5)

    start_button = tk.Button(root, text="Start Sniffing", command=start_sniffing)
    start_button.pack(pady=20)

    global text_box  # Declare the global variable for text_box
    text_box = scrolledtext.ScrolledText(root, width=80, height=20)
    text_box.pack(padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
