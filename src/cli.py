from PIL import Image
import pyfiglet

ASCII_CHARS = " .:=-+*%@,/?'\"`~|()&^$"
ansi = lambda r, g, b: f"\033[38;2;{r};{g};{b}m"

def get_large_text_lines():
    h_lines = pyfiglet.figlet_format("Harness", font="doom").splitlines()
    d_lines = pyfiglet.figlet_format(".", font="doom").splitlines()
    active = [i for i, (h, d) in enumerate(zip(h_lines, d_lines)) if h.strip() or d.strip()]
    if not active:
        return [], 0
    h_lines, d_lines = h_lines[active[0]:active[-1]+1], d_lines[active[0]:active[-1]+1]
    h_max, d_max = max(len(l) for l in h_lines), max(len(l) for l in d_lines)
    combined = []
    for hl, dl in zip(h_lines, d_lines):
        line = ""
        for i, char in enumerate(hl.ljust(h_max)):
            if char == ' ':
                line += " "
            else:
                ratio = i / max(1, h_max - 1)
                r, g, b = int(180 - 92 * ratio), int(86 * ratio), int(80 + 86 * ratio)
                line += f"{ansi(r, g, b)}{char}"
        combined.append(f"{line}\033[38;2;255;255;255m{dl.ljust(d_max)}\033[0m")
    return combined, h_max + d_max

def render_terminal_ui(image_path="/home/ranta_unix/projects/Harness/together.png", total_width=110):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    orig_w, orig_h = img.size
    blackhole_img = img.crop((0, 0, int(orig_w * 0.55), orig_h))

    aspect_ratio = orig_h / (orig_w * 0.55)
    bh_width = int(total_width * 0.52)
    text_lines, text_width = get_large_text_lines()
    bh_height = max(int(aspect_ratio * bh_width * 0.5), len(text_lines) + 6)

    blackhole_img = blackhole_img.resize((bh_width, bh_height))
    pixels = blackhole_img.load()

    white, reset = "\033[38;2;255;255;255m", "\033[0m"
    print(f"{white}╭{'─' * (total_width + 2)}╮{reset}")
    text_start_y = (bh_height - len(text_lines)) // 2

    for y in range(bh_height):
        line_chars = ""

        for x in range(bh_width):
            r, g, b = pixels[x, y]
            brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
            char_idx = int((brightness / 255) * (len(ASCII_CHARS) - 1))
            line_chars += f"{ansi(r, g, b)}{ASCII_CHARS[char_idx]}"

        right_side_spaces = total_width - bh_width

        if text_start_y <= y < text_start_y + len(text_lines):
            font_row = y - text_start_y
            padding_left = max(0, (right_side_spaces - text_width) // 2)
            padding_right = max(0, right_side_spaces - text_width - padding_left)
            line_chars += " " * padding_left + text_lines[font_row] + " " * padding_right
        else:
            line_chars += " " * right_side_spaces

        print(f"{white}│ {reset}{line_chars}{white} │{reset}")

    print(f"{white}╰{'─' * (total_width + 2)}╯{reset}")

if __name__ == "__main__":
    render_terminal_ui()