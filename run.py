#!/usr/bin/env python3
"""
Book Translator - Desktop Application
======================================
Launch the Book Translator as a desktop app with system tray support.

Usage:
    python run.py
    python -m book_translator
"""
import sys
import os
import threading
import time
import webbrowser
from pathlib import Path

# Add package to path if running directly
package_dir = Path(__file__).parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

# Determine if we're running as a packaged .exe
if getattr(sys, 'frozen', False):
    BUNDLE_DIR = sys._MEIPASS
    APP_DIR = os.path.dirname(sys.executable)
else:
    BUNDLE_DIR = str(package_dir)
    APP_DIR = BUNDLE_DIR

# Set environment for config
os.environ['BOOK_TRANSLATOR_APP_DIR'] = APP_DIR
os.environ['BOOK_TRANSLATOR_BUNDLE_DIR'] = BUNDLE_DIR

# Change to app directory
os.chdir(APP_DIR)

# Create necessary folders
for folder in ['uploads', 'translations', 'logs']:
    os.makedirs(os.path.join(APP_DIR, folder), exist_ok=True)

# Import pystray for system tray
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_PYSTRAY = True
except ImportError:
    HAS_PYSTRAY = False

# Colors for terminal output
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'

# Global variables
flask_thread = None
tray_icon = None
server_port = 5001
APP_URL = f'http://localhost:{server_port}'


def check_ollama():
    """Check if Ollama is running"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def print_banner():
    """Display startup banner"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  📚 BOOK TRANSLATOR - Desktop Edition v2.0{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"  Server: {APP_URL}")
    print(f"  Working Directory: {os.getcwd()}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")


def create_tray_icon_image():
    """Create icon for the system tray"""
    size = 64
    image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Book shape (blue background)
    draw.rounded_rectangle([4, 4, 60, 60], radius=8, fill=(41, 128, 185, 255))
    
    # Inner area (lighter blue)
    draw.rounded_rectangle([8, 8, 56, 56], radius=6, fill=(52, 152, 219, 255))
    
    # Pages (white)
    draw.rounded_rectangle([14, 12, 52, 52], radius=4, fill=(255, 255, 255, 255))
    
    # Spine (darker blue)
    draw.rounded_rectangle([4, 4, 16, 60], radius=4, fill=(31, 97, 141, 255))
    
    # Text lines
    for y in [20, 30, 40]:
        draw.rounded_rectangle([20, y, 48, y + 4], radius=2, fill=(189, 195, 199, 255))
    
    return image


def start_flask_server():
    """Start Flask server in a separate thread"""
    print(f"{Colors.GREEN}🚀 Starting Flask server on port {server_port}...{Colors.RESET}")
    
    # Import and run Flask app
    from book_translator.app import create_app
    app = create_app()
    app.run(host='127.0.0.1', port=server_port, debug=False, use_reloader=False, threaded=True)


def open_app_window(icon=None, item=None):
    """Open the translator in the default browser"""
    print(f"{Colors.CYAN}🌐 Opening application in browser...{Colors.RESET}")
    try:
        webbrowser.open(APP_URL)
    except Exception as e:
        print(f"{Colors.RED}✗ Could not open browser: {e}{Colors.RESET}")


def quit_app(icon=None, item=None):
    """Completely quit the application"""
    print(f"\n{Colors.YELLOW}👋 Shutting down Book Translator...{Colors.RESET}")
    if tray_icon:
        tray_icon.stop()
    os._exit(0)


def create_tray_menu():
    """Create the system tray menu"""
    return pystray.Menu(
        pystray.MenuItem("📖 Open Book Translator", open_app_window, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(f"🌐 Server: localhost:{server_port}", lambda: None, enabled=False),
        pystray.MenuItem("✅ Server Running", lambda: None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("❌ Quit Completely", quit_app)
    )


def run_with_tray():
    """Run the app with system tray support"""
    global flask_thread, tray_icon
    
    print(f"{Colors.GREEN}🚀 Starting in System Tray mode...{Colors.RESET}")
    print(f"{Colors.CYAN}   The server runs in background.{Colors.RESET}")
    print(f"{Colors.CYAN}   Close browser tab anytime - reopen from tray icon.{Colors.RESET}")
    print(f"{Colors.YELLOW}   To fully quit: right-click tray icon → Quit Completely{Colors.RESET}\n")
    
    # Start Flask server in background thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    
    # Give server time to start
    print(f"{Colors.YELLOW}⏳ Starting server...{Colors.RESET}")
    time.sleep(2)
    
    # Open browser window automatically
    print(f"{Colors.GREEN}✓ Server started!{Colors.RESET}")
    open_app_window()
    
    # Create and run system tray icon
    icon_image = create_tray_icon_image()
    tray_icon = pystray.Icon(
        "BookTranslator",
        icon_image,
        "Book Translator - Click to Open",
        menu=create_tray_menu()
    )
    
    print(f"\n{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}✓ Book Translator is running!{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")
    print(f"  🌐 URL: {APP_URL}")
    print(f"  📌 System tray icon active")
    print(f"  💡 Close browser anytime - server keeps running")
    print(f"  🔄 Click tray icon to reopen")
    print(f"{Colors.GREEN}{'='*60}{Colors.RESET}\n")
    
    # Run the tray icon (this blocks but keeps the app alive)
    tray_icon.run()


def run_simple():
    """Run without tray (fallback)"""
    from book_translator.app import create_app
    
    print(f"{Colors.GREEN}🚀 Starting Flask server...{Colors.RESET}")
    print(f"{Colors.CYAN}   URL: {APP_URL}{Colors.RESET}")
    print(f"\n{Colors.RED}   Press Ctrl+C to close{Colors.RESET}\n")
    
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(APP_URL)
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    app = create_app()
    app.run(host='127.0.0.1', port=server_port, debug=False, use_reloader=False, threaded=True)


def main():
    """Main entry point"""
    print_banner()
    
    # Check Ollama
    print(f"{Colors.YELLOW}🔍 Checking Ollama...{Colors.RESET}")
    if check_ollama():
        print(f"{Colors.GREEN}   ✓ Ollama is running{Colors.RESET}")
    else:
        print(f"{Colors.RED}   ⚠️  Ollama not detected at localhost:11434{Colors.RESET}")
        print(f"{Colors.YELLOW}   Please start Ollama before translating{Colors.RESET}")
    
    print()
    
    if HAS_PYSTRAY:
        run_with_tray()
    else:
        print(f"{Colors.YELLOW}⚠ pystray not installed, running without tray{Colors.RESET}")
        run_simple()


if __name__ == '__main__':
    main()
