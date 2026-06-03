import time
import sys

def animate_spinner():
    spinner = ['|', '/', '-', '\\']
    i = 0
    
    try:
        while True:
            # \033[H moves cursor home, \033[J clears the screen
            sys.stdout.write(f"\033[H\033[J")
            sys.stdout.write(f"Animating in Terminal: {spinner[i % 4]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nAnimation stopped.")

if __name__ == "__main__":
    animate_spinner()