# -*- coding: utf-8 -*-

import PyInstaller.__main__

PyInstaller.__main__.run([
    'main.py',
    '--name=RR4_AutoClicker',
    '--onefile',
    '--windowed',
    '--noconfirm',
    '--clean'
])