# -*- coding: utf-8 -*-
"""
Русская Рыбалка 4 - Автокликер v3.0
Всё в одном файле для корректной сборки
"""

import sys
import os
import json
import threading
import time
import cv2
import numpy as np
import mss
import keyboard
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QTabWidget, QFrame, QStatusBar,
    QFileDialog, QMessageBox, QComboBox, QTableWidget, QTableWidgetItem,
    QDialog, QFormLayout, QDialogButtonBox, QHeaderView
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController


# === ОПРЕДЕЛЕНИЕ ПУТИ ===
if getattr(sys, 'frozen', False):
    CURRENT_DIR = os.path.dirname(sys.executable)
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)


# === СТИЛИ ===
DARK_STYLE = """
QMainWindow { background-color: #1e1e2e; }
QWidget { background-color: #1e1e2e; color: #cdd6f4; font-family: 'Segoe UI', Arial; font-size: 10pt; }
QGroupBox { border: 2px solid #45475a; border-radius: 8px; margin-top: 12px; padding-top: 10px; font-weight: bold; color: #89b4fa; }
QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 8px; color: #89b4fa; }
QPushButton { background-color: #45475a; border: none; border-radius: 6px; padding: 8px 16px; color: #cdd6f4; font-weight: bold; }
QPushButton:hover { background-color: #585b70; }
QPushButton:pressed { background-color: #313244; }
QPushButton:checked { background-color: #4CAF50; color: white; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #313244; border: 2px solid #45475a; border-radius: 6px; padding: 6px 10px; color: #cdd6f4; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #89b4fa; }
QCheckBox { spacing: 8px; color: #cdd6f4; }
QCheckBox::indicator { width: 20px; height: 20px; border-radius: 4px; border: 2px solid #45475a; background-color: #313244; }
QCheckBox::indicator:checked { background-color: #4CAF50; border-color: #4CAF50; }
QTabWidget::pane { border: 2px solid #45475a; border-radius: 8px; background-color: #1e1e2e; }
QTabBar::tab { background-color: #313244; border: none; padding: 10px 20px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; color: #cdd6f4; }
QTabBar::tab:selected { background-color: #45475a; color: #89b4fa; }
QStatusBar { background-color: #313244; color: #a6adc8; border-top: 1px solid #45475a; }
"""


# === ДВИЖОК КЛИКЕРА ===
class ClickerEngine(QObject):
    bite_detected = pyqtSignal()
    tension_critical = pyqtSignal()
    fish_caught = pyqtSignal()
    status_changed = pyqtSignal(str)
    hotkey_pressed = pyqtSignal(str)
    countdown_update = pyqtSignal(float)
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.mouse = MouseController()
        self.keyboard_ctrl = KeyboardController()
        self.hotkey_hooks = []
        self.hotkey_status = "Горячие клавиши не настроены"
        
        self.action_states = {
            'light_pilking': False,
            'heavy_pilking': False,
            'reel_in': False,
            'turn_left': False,
            'turn_right': False
        }
        
        self.threads = {}
        self.running = True
        self.detection_active = False
        self.auto_reel_enabled = False
        self.tension_release_enabled = False
        self.is_reeling = False
        self.bite_was_detected = False
        
        self.bite_template = None
        self.bite_template_mask = None
        self.load_templates()
        self.setup_hotkeys()
        
    def load_templates(self):
        icon_config = self.config.get('bite_icon', {})
        icon_path = icon_config.get('path', 'bite_icon.png')
        self.bite_template, self.bite_template_mask = self._load_template(icon_path)
        if self.bite_template is not None:
            self.status_changed.emit("✓ Ионка загружена")
        else:
            self.status_changed.emit(f"✗ Иконка не найдена: {icon_path}")
        
    def _load_template(self, filename):
        paths = [
            os.path.join(CURRENT_DIR, filename),
            filename
        ]
        for path in paths:
            if os.path.exists(path):
                template = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if template is None:
                    continue

                if len(template.shape) == 2:
                    return template, None

                if template.shape[2] == 4:
                    alpha = template[:, :, 3]
                    mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)[1]
                    # Если маска пустая/почти пустая, отключаем masked matching,
                    # чтобы избежать NaN/-inf в cv2.matchTemplate.
                    if cv2.countNonZero(mask) < 10:
                        mask = None
                    bgr = template[:, :, :3]
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    return gray, mask

                gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                return gray, None
        return None, None

    def match_bite_icon(self, screenshot_bgra):
        if self.bite_template is None:
            return 0.0

        gray = cv2.cvtColor(screenshot_bgra, cv2.COLOR_BGRA2GRAY)
        icon_config = self.config.get('bite_icon', {})
        match_method = icon_config.get('match_method', 'classic')

        use_masked = match_method == 'masked' and self.bite_template_mask is not None

        if use_masked:
            result = cv2.matchTemplate(
                gray,
                self.bite_template,
                cv2.TM_CCORR_NORMED,
                mask=self.bite_template_mask
            )
        else:
            result = cv2.matchTemplate(gray, self.bite_template, cv2.TM_CCOEFF_NORMED)

        _, max_val, _, _ = cv2.minMaxLoc(result)
        max_val = float(max_val)
        if not np.isfinite(max_val):
            return 0.0
        return max(0.0, min(1.0, max_val))

    def normalize_capture_region(self, left, top, width, height):
        """Гарантируем, что область захвата полностью внутри выбранного разрешения."""
        screen_cfg = self.config.get('screen_resolution', {})
        screen_w = int(screen_cfg.get('width', 2560))
        screen_h = int(screen_cfg.get('height', 1440))

        width = max(1, min(int(width), screen_w))
        height = max(1, min(int(height), screen_h))

        max_left = max(0, screen_w - width)
        max_top = max(0, screen_h - height)

        left = max(0, min(int(left), max_left))
        top = max(0, min(int(top), max_top))

        return {
            'left': left,
            'top': top,
            'width': width,
            'height': height
        }

    def confirm_bite_signal(self, sct, search_region, threshold, confirm_frames=3, frame_delay=0.03):
        """Подтверждаем поклёвку только если несколько кадров подряд >= порога."""
        confirmations = 0
        for _ in range(max(1, confirm_frames)):
            screenshot = sct.grab(search_region)
            img = np.array(screenshot)
            score = self.match_bite_icon(img)
            if score >= threshold:
                confirmations += 1
            if frame_delay > 0:
                time.sleep(frame_delay)
        return confirmations >= max(1, confirm_frames)
        
    def setup_hotkeys(self):
        for hook in self.hotkey_hooks:
            try:
                keyboard.unhook(hook)
            except Exception:
                pass
        self.hotkey_hooks.clear()

        hotkeys = self.config.get('hotkeys', {})
        registered = []
        errors = []

        for action, key in hotkeys.items():
            if not key:
                continue
            try:
                hook = keyboard.on_press_key(
                    str(key).strip().lower(),
                    lambda event, current_action=action: self.on_hotkey(current_action),
                    suppress=False
                )
                self.hotkey_hooks.append(hook)
                registered.append(f"{action}: {key}")
            except Exception as error:
                errors.append(f"{key}: {error}")

        self.hotkey_status = (
            f"Горячие клавиши зарегистрированы: {len(registered)}"
            if not errors else
            f"Зарегистрировано: {len(registered)}; ошибки: {'; '.join(errors)}"
        )
        if not registered and not errors:
            self.hotkey_status = "В config.json не заданы горячие клавиши"

    def on_hotkey(self, action):
        self.hotkey_pressed.emit(action)
        
    def set_action_state(self, action_key, enabled):
        self.action_states[action_key] = enabled
        if enabled:
            self.start_action(action_key)
        else:
            self.stop_action(action_key)
            
    def start_action(self, action_key):
        if action_key in self.threads and self.threads[action_key].is_alive():
            return
        thread = None
        if action_key == 'light_pilking':
            thread = threading.Thread(target=self.pilking_loop, args=('light',), daemon=True)
        elif action_key == 'heavy_pilking':
            thread = threading.Thread(target=self.pilking_loop, args=('heavy',), daemon=True)
        elif action_key == 'reel_in':
            thread = threading.Thread(target=self.reel_in_loop, daemon=True)
        elif action_key == 'turn_left':
            thread = threading.Thread(target=self.turn_loop, args=('left',), daemon=True)
        elif action_key == 'turn_right':
            thread = threading.Thread(target=self.turn_loop, args=('right',), daemon=True)
        if thread:
            self.threads[action_key] = thread
            thread.start()
        
    def stop_action(self, action_key):
        self.action_states[action_key] = False
        
    def pilking_loop(self, pilking_type):
        action_key = f'{pilking_type}_pilking'
        presets = self.config.get('presets', {})
        preset = presets.get(f'{pilking_type}_pilking', {})
        hold_time = preset.get('hold_time', 0.3 if pilking_type == 'light' else 0.5)
        wait_time = preset.get('wait_time', 2.1 if pilking_type == 'light' else 2.3)
        
        while self.running and self.action_states.get(action_key, False):
            if self.is_reeling:
                time.sleep(0.1)
                continue
            self.mouse.press(Button.right)
            time.sleep(hold_time)
            self.mouse.release(Button.right)
            time.sleep(wait_time)
            
    def reel_in_loop(self):
        self.is_reeling = True
        self.mouse.press(Button.left)
        self.mouse.press(Button.middle)
        self.mouse.press(Button.right)  # Добавлено ПКМ
        while self.running and self.action_states.get('reel_in', False):
            time.sleep(0.02)
        self.mouse.release(Button.left)
        self.mouse.release(Button.middle)
        self.mouse.release(Button.right)  # Отпускаем ПКМ
        self.is_reeling = False
        
    def turn_loop(self, direction):
        action_key = f'turn_{direction}'
        key = 'a' if direction == 'left' else 'd'
        while self.running and self.action_states.get(action_key, False):
            self.keyboard_ctrl.press(key)
            time.sleep(0.05)
        self.keyboard_ctrl.release(key)
        
    def start_detection(self, auto_reel=True, tension_release=True):
        self.detection_active = True
        self.auto_reel_enabled = auto_reel
        self.tension_release_enabled = tension_release
        self.bite_was_detected = False
        thread = threading.Thread(target=self.fishing_cycle, daemon=True)
        thread.start()
        self.status_changed.emit("🎣 АВТО-ЛОВЛЯ ЗАПУЩЕНА")
        
    def stop_detection(self):
        self.detection_active = False
        self.bite_was_detected = False
        self.set_action_state('light_pilking', False)
        self.set_action_state('heavy_pilking', False)
        self.set_action_state('reel_in', False)
        self.status_changed.emit("⏹ Остановлено")
        
    def fishing_cycle(self):
        fishing_config = self.config.get('auto_fishing', {})
        wait_after_cast = fishing_config.get('wait_after_cast', 32)
        cast_delay = fishing_config.get('cast_delay', 1.0)
        after_catch_delay = fishing_config.get('after_catch_delay', 0.5)
        pilking_type = fishing_config.get('pilking_type', 'light')
        
        icon_config = self.config.get('bite_icon', {})
        tension_config = self.config.get('tension_bar', {})
        
        region = icon_config.get('region', {})
        bite_threshold = icon_config.get('threshold', 0.8)
        bite_confirm_frames = icon_config.get('confirm_frames', 3)
        bite_confirm_delay = icon_config.get('confirm_delay', 0.03)
        search_region = self.normalize_capture_region(
            region.get('x', 804),
            region.get('y', 1318),
            region.get('width', 133),
            region.get('height', 133)
        )
        
        tension_x = tension_config.get('check_x', 900)
        tension_y = tension_config.get('check_y', 1412)
        r_min = tension_config.get('color_r_min', 180)
        
        while self.running and self.detection_active:
            try:
                # === ЭТАП 1: Ожидание с отслеживанием поклёвки ===
                self.status_changed.emit(f"⏳ Ожидание {wait_after_cast} сек (отслеживание вкл)...")
                
                bite_during_wait = False
                wait_ticks = int(round(float(wait_after_cast) * 20))  # 20 проверок в секунду
                
                with mss.mss() as sct:
                    for tick in range(wait_ticks):
                        if not self.detection_active:
                            return
                        
                        # Обновляем таймер каждую секунду
                        if tick % 20 == 0:
                            self.countdown_update.emit(max(0.0, round(float(wait_after_cast) - tick / 20, 1)))
                        
                        # Проверяем поклёвку во время ожидания
                        if self.bite_template is not None:
                            screenshot = sct.grab(search_region)
                            img = np.array(screenshot)
                            max_val = self.match_bite_icon(img)
                            
                            if max_val >= bite_threshold and self.confirm_bite_signal(
                                sct,
                                search_region,
                                bite_threshold,
                                confirm_frames=bite_confirm_frames,
                                frame_delay=bite_confirm_delay
                            ):
                                # Поклёвка во время спуска!
                                bite_during_wait = True
                                self.bite_detected.emit()
                                self.status_changed.emit(f"🐟 ПОКЛЁВКА при спуске! ({max_val:.0%})")
                                self.set_action_state('reel_in', True)
                                break
                        
                        time.sleep(0.05)
                    
                    if not self.detection_active:
                        return
                    
                    # Если поклёвка была при спуске - ждём вытаскивания
                    if bite_during_wait:
                        self.bite_was_detected = True
                        no_bite_counter = 0
                        tension_cooldown = 0
                        
                        while self.running and self.detection_active:
                            screenshot = sct.grab(search_region)
                            img = np.array(screenshot)
                            max_val = self.match_bite_icon(img)
                            
                            if max_val >= bite_threshold:
                                no_bite_counter = 0
                            else:
                                no_bite_counter += 1
                                if no_bite_counter >= 8:
                                    # Рыба поймана
                                    self.fish_caught.emit()
                                    self.status_changed.emit("✅ Рыба поймана!")
                                    self.set_action_state('reel_in', False)
                                    self.bite_was_detected = False
                                    time.sleep(after_catch_delay)
                                    self.keyboard_ctrl.press(' ')
                                    time.sleep(0.1)
                                    self.keyboard_ctrl.release(' ')
                                    self.status_changed.emit("🎣 Забрасываем...")
                                    time.sleep(cast_delay)
                                    self.mouse.click(Button.left)
                                    break
                            
                            # Проверка натяжения
                            if self.tension_release_enabled and tension_cooldown <= 0 and self.is_reeling:
                                tp = {'left': tension_x, 'top': tension_y, 'width': 1, 'height': 1}
                                pixel = sct.grab(tp)
                                pd = np.array(pixel)[0][0]
                                if pd[2] >= r_min and pd[1] < 100 and pd[0] < 100:
                                    self.mouse.scroll(0, -1)
                                    time.sleep(0.05)
                                    self.mouse.scroll(0, 1)
                                    tension_cooldown = 15
                            
                            if tension_cooldown > 0:
                                tension_cooldown -= 1
                            time.sleep(0.05)
                        
                        continue  # Следующий цикл с начала
                    
                    # === ЭТАП 2: Таймер истёк — закрыть катушку + пилкинг ===
                    self.status_changed.emit("🔒 Закрываем катушку...")
                    self.mouse.click(Button.left)
                    time.sleep(0.3)
                    
                    self.status_changed.emit(f"🔄 Пилкинг ({pilking_type})...")
                    self.set_action_state(f'{pilking_type}_pilking', True)
                    
                    # === ЭТАП 3: Ждём поклёвку с пилкингом ===
                    self.bite_was_detected = False
                    no_bite_counter = 0
                    tension_cooldown = 0
                    
                    while self.running and self.detection_active:
                        if self.bite_template is not None:
                            screenshot = sct.grab(search_region)
                            img = np.array(screenshot)
                            max_val = self.match_bite_icon(img)
                            
                            if max_val >= bite_threshold:
                                no_bite_counter = 0
                                if not self.bite_was_detected:
                                    if self.confirm_bite_signal(
                                        sct,
                                        search_region,
                                        bite_threshold,
                                        confirm_frames=bite_confirm_frames,
                                        frame_delay=bite_confirm_delay
                                    ):
                                        self.bite_was_detected = True
                                        self.bite_detected.emit()
                                        self.status_changed.emit(f"🐟 ПОКЛЁВКА! ({max_val:.0%})")
                                        self.set_action_state('light_pilking', False)
                                        self.set_action_state('heavy_pilking', False)
                                        self.set_action_state('reel_in', True)
                            else:
                                if self.bite_was_detected:
                                    no_bite_counter += 1
                                    if no_bite_counter >= 8:
                                        self.fish_caught.emit()
                                        self.status_changed.emit("✅ Рыба поймана!")
                                        self.set_action_state('reel_in', False)
                                        self.bite_was_detected = False
                                        time.sleep(after_catch_delay)
                                        self.keyboard_ctrl.press(' ')
                                        time.sleep(0.1)
                                        self.keyboard_ctrl.release(' ')
                                        self.status_changed.emit("🎣 Забрасываем...")
                                        time.sleep(cast_delay)
                                        self.mouse.click(Button.left)
                                        break
                        
                        if self.tension_release_enabled and tension_cooldown <= 0 and self.is_reeling:
                            tp = {'left': tension_x, 'top': tension_y, 'width': 1, 'height': 1}
                            pixel = sct.grab(tp)
                            pd = np.array(pixel)[0][0]
                            if pd[2] >= r_min and pd[1] < 100 and pd[0] < 100:
                                self.mouse.scroll(0, -1)
                                time.sleep(0.05)
                                self.mouse.scroll(0, 1)
                                tension_cooldown = 15
                        
                        if tension_cooldown > 0:
                            tension_cooldown -= 1
                        time.sleep(0.05)
                        
            except Exception as e:
                self.status_changed.emit(f"Ошибка: {str(e)}")
                time.sleep(1)
                
    def test_icon_detection(self, icon_path, threshold, region):
        try:
            max_val = self.get_icon_match_score(icon_path, region)
            if max_val is None:
                return f"❌ Не найден: {icon_path}"
            status = "✅ НАЙДЕНО" if max_val >= threshold else "❌ Не найдено"
            return f"{status}\nСовпадение: {max_val:.1%}"
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"

    def get_icon_match_score(self, icon_path, region):
        template, mask = self._load_template(icon_path)
        if template is None:
            return None

        search_region = self.normalize_capture_region(region[0], region[1], region[2], region[3])
        with mss.mss() as sct:
            screenshot = sct.grab(search_region)
            img = np.array(screenshot)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            icon_config = self.config.get('bite_icon', {})
            match_method = icon_config.get('match_method', 'classic')
            use_masked = match_method == 'masked' and mask is not None

            if use_masked:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCORR_NORMED, mask=mask)
            else:
                result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

            _, max_val, _, _ = cv2.minMaxLoc(result)
            max_val = float(max_val)
            if not np.isfinite(max_val):
                return 0.0
            return max(0.0, min(1.0, max_val))
            
    def test_tension_point(self, x, y):
        try:
            with mss.mss() as sct:
                region = {'left': x, 'top': y, 'width': 1, 'height': 1}
                pixel = sct.grab(region)
                pd = np.array(pixel)[0][0]
                is_red = pd[2] >= 180 and pd[1] < 100 and pd[0] < 100
                status = "🔴 КРАСНЫЙ" if is_red else "⚪ Не красный"
                return f"{status}\nRGB: ({pd[2]}, {pd[1]}, {pd[0]})"
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"
            
    def update_config(self, config):
        self.config = config
        self.load_templates()
        
    def update_hotkeys(self, hotkeys):
        self.config['hotkeys'] = hotkeys
        self.setup_hotkeys()
        
    def stop_all(self):
        self.running = False
        self.detection_active = False
        for k in self.action_states:
            self.action_states[k] = False
        try:
            self.mouse.release(Button.left)
            self.mouse.release(Button.middle)
            self.mouse.release(Button.right)
            self.keyboard_ctrl.release('a')
            self.keyboard_ctrl.release('d')
        except:
            pass
        for hook in self.hotkey_hooks:
            try:
                keyboard.unhook(hook)
            except Exception:
                pass
        self.hotkey_hooks.clear()


# === ВИДЖЕТЫ ===
class StatusIndicator(QFrame):
    def __init__(self):
        super().__init__()
        self.setFixedSize(20, 20)
        self.active = False
        self.update_style()
        
    def set_active(self, active):
        self.active = active
        self.update_style()
        
    def update_style(self):
        color = "#4CAF50" if self.active else "#666666"
        self.setStyleSheet(f"QFrame {{ background-color: {color}; border-radius: 10px; border: 2px solid #333; }}")


class HotkeyButton(QPushButton):
    keyChanged = pyqtSignal(str)
    
    def __init__(self, current_key=""):
        super().__init__()
        self.current_key = current_key
        self.waiting = False
        self.update_text()
        self.clicked.connect(self.start_capture)
        
    def update_text(self):
        self.setText("..." if self.waiting else (self.current_key.upper() if self.current_key else "—"))
    
    def start_capture(self):
        self.waiting = True
        self.update_text()
        self.setFocus()
        
    def keyPressEvent(self, event):
        if self.waiting:
            key = event.text().lower()
            if key:
                self.current_key = key
                self.keyChanged.emit(key)
            self.waiting = False
            self.update_text()
        else:
            super().keyPressEvent(event)
            
    def focusOutEvent(self, event):
        self.waiting = False
        self.update_text()
        super().focusOutEvent(event)


class BankPresetDialog(QDialog):
    def __init__(self, parent=None, bank="", bait="", duration=20.5, pilking_type="light"):
        super().__init__(parent)
        self.setWindowTitle("Пресет банки")
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.bank_edit = QLineEdit(str(bank))
        self.bait_edit = QLineEdit(str(bait))
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0.5, 600.0)
        self.time_spin.setDecimals(1)
        self.time_spin.setSingleStep(0.5)
        self.time_spin.setValue(float(duration))
        self.time_spin.setSuffix(" с")
        self.pilking_combo = QComboBox()
        self.pilking_combo.addItem("Лёгкий", "light")
        self.pilking_combo.addItem("Сильный", "heavy")
        pilking_index = self.pilking_combo.findData(pilking_type)
        self.pilking_combo.setCurrentIndex(max(0, pilking_index))
        form.addRow("Номер банки/ямы:", self.bank_edit)
        form.addRow("Приманка:", self.bait_edit)
        form.addRow("Время:", self.time_spin)
        form.addRow("Пилкинг:", self.pilking_combo)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self):
        return {
            "bank": self.bank_edit.text().strip(),
            "bait": self.bait_edit.text().strip(),
            "time": self.time_spin.value(),
            "pilking_type": self.pilking_combo.currentData()
        }


# === ГЛАВНОЕ ОКНО ===
class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.engine = None
        self.action_states = {k: False for k in ['light_pilking', 'heavy_pilking', 'reel_in', 'turn_left', 'turn_right']}
        self.indicators = {}
        self.action_buttons = {}
        self.monitor_presets = {}
        self.config.setdefault('bank_presets', [{'bank': '50', 'bait': 'Гигантский виброхвост 600гр', 'time': 20.5}])
        self.config.setdefault('selected_bank_index', 0)
        self.bank_presets = self.config.get('bank_presets', [])
        for bank_preset in self.bank_presets:
            bank_preset.setdefault("pilking_type", "light")
        
        self.init_ui()
        self.init_engine()
        self.load_settings()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)
        
    def init_ui(self):
        self.setWindowTitle("РР4 Автокликер v3.0")
        self.setMinimumSize(650, 600)
        self.setStyleSheet(DARK_STYLE)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        header = QLabel("🎣 Русская Рыбалка 4 - Автокликер")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("color: #4FC3F7; padding: 10px;")
        layout.addWidget(header)
        
        tabs = QTabWidget()
        tabs.addTab(self.create_main_tab(), "Главная")
        tabs.addTab(self.create_settings_tab(), "Настройки")
        tabs.addTab(self.create_detection_tab(), "Распознавание")
        tabs.addTab(self.create_banks_tab(), "Банки")
        layout.addWidget(tabs)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        
    def create_main_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Ручное управление
        group1 = QGroupBox("Ручное управление")
        g1_layout = QGridLayout(group1)
        
        actions = [
            ('light_pilking', 'Лёгкий пилкинг', 'R'),
            ('heavy_pilking', 'Сильный пилкинг', 'U'),
            ('reel_in', 'ЛКМ+СКМ', '0'),
            ('turn_left', 'Влево (A)', '7'),
            ('turn_right', 'Вправо (D)', '8'),
        ]
        
        for i, (key, name, hk) in enumerate(actions):
            ind = StatusIndicator()
            self.indicators[key] = ind
            g1_layout.addWidget(ind, i, 0)
            g1_layout.addWidget(QLabel(name), i, 1)
            lbl = QLabel(f"[{hk}]")
            lbl.setStyleSheet("color: #FFB74D; font-weight: bold;")
            g1_layout.addWidget(lbl, i, 2)
            btn = QPushButton("ВКЛ")
            btn.setCheckable(True)
            btn.setFixedWidth(70)
            btn.clicked.connect(lambda c, k=key: self.toggle_action(k, c))
            self.action_buttons[key] = btn
            g1_layout.addWidget(btn, i, 3)
            
        layout.addWidget(group1)
        
        # Автоловля
        group2 = QGroupBox("🤖 Автоматическая ловля")
        g2_layout = QVBoxLayout(group2)
        
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Ожидание:"))
        self.spin_wait = QDoubleSpinBox()
        self.spin_wait.setRange(0.5, 600.0)
        self.spin_wait.setDecimals(1)
        self.spin_wait.setSingleStep(0.5)
        self.spin_wait.setValue(32.0)
        self.spin_wait.setSuffix(" сек")
        row1.addWidget(self.spin_wait)
        row1.addWidget(QLabel("Пилкинг:"))
        self.combo_pilking = QComboBox()
        self.combo_pilking.addItems(["Лёгкий", "Сильный"])
        row1.addWidget(self.combo_pilking)
        row1.addStretch()
        g2_layout.addLayout(row1)
        
        self.chk_reel = QCheckBox("Авто-подсечка")
        self.chk_reel.setChecked(True)
        self.chk_tension = QCheckBox("Авто-сброс натяжения")
        self.chk_tension.setChecked(True)
        g2_layout.addWidget(self.chk_reel)
        g2_layout.addWidget(self.chk_tension)
        
        self.lbl_countdown = QLabel("")
        self.lbl_countdown.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.lbl_countdown.setStyleSheet("color: #FFB74D;")
        self.lbl_countdown.setAlignment(Qt.AlignCenter)
        g2_layout.addWidget(self.lbl_countdown)
        
        self.lbl_status = QLabel("Готов")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        g2_layout.addWidget(self.lbl_status)
        
        self.btn_start = QPushButton("▶ ЗАПУСТИТЬ АВТО-ЛОВЛЮ")
        self.btn_start.setCheckable(True)
        self.btn_start.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.btn_start.setMinimumHeight(50)
        self.btn_start.clicked.connect(self.toggle_auto)
        g2_layout.addWidget(self.btn_start)
        
        layout.addWidget(group2)
        layout.addStretch()
        return tab
    
    def create_banks_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        info = QLabel("Сохраняй свои банки, приманки и время ожидания. Выбранный пресет автоматически задаёт время для автоматической ловли.")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.bank_table = QTableWidget(0, 3)
        self.bank_table.setHorizontalHeaderLabels(["Номер банки/ямы", "Приманка", "Время"])
        self.bank_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.bank_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.bank_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.bank_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.bank_table.setSelectionMode(QTableWidget.SingleSelection)
        self.bank_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.bank_table.doubleClicked.connect(self.edit_bank)
        layout.addWidget(self.bank_table)
        buttons = QHBoxLayout()
        self.btn_bank_add = QPushButton("＋ Добавить")
        self.btn_bank_edit = QPushButton("✎ Изменить")
        self.btn_bank_delete = QPushButton("✕ Удалить")
        self.btn_bank_add.clicked.connect(self.add_bank)
        self.btn_bank_edit.clicked.connect(self.edit_bank)
        self.btn_bank_delete.clicked.connect(self.delete_bank)
        buttons.addWidget(self.btn_bank_add)
        buttons.addWidget(self.btn_bank_edit)
        buttons.addWidget(self.btn_bank_delete)
        layout.addLayout(buttons)
        active_row = QHBoxLayout()
        active_row.addWidget(QLabel("Активная банка:"))
        self.bank_combo = QComboBox()
        self.bank_combo.currentIndexChanged.connect(self.on_bank_selected)
        active_row.addWidget(self.bank_combo, 1)
        layout.addLayout(active_row)
        self.refresh_bank_table()
        self.refresh_bank_combo()
        layout.addStretch()
        return tab

    def refresh_bank_table(self):
        if not hasattr(self, "bank_table"):
            return
        self.bank_table.setRowCount(0)
        for preset in self.bank_presets:
            row = self.bank_table.rowCount()
            self.bank_table.insertRow(row)
            values = [str(preset.get("bank", "")), str(preset.get("bait", "")), f'{float(preset.get("time", 0.5)):g} с']
            for column, value in enumerate(values):
                self.bank_table.setItem(row, column, QTableWidgetItem(value))

    def refresh_bank_combo(self):
        if not hasattr(self, "bank_combo"):
            return
        self.bank_combo.blockSignals(True)
        self.bank_combo.clear()
        for preset in self.bank_presets:
            pilking_label = "Сильный" if preset.get("pilking_type", "light") == "heavy" else "Лёгкий"
            label = f'{preset.get("bank", "")} — {preset.get("bait", "")} ({float(preset.get("time", 0.5)):g} с, {pilking_label})'
            self.bank_combo.addItem(label)
        selected = int(self.config.get("selected_bank_index", 0))
        if self.bank_presets:
            selected = max(0, min(selected, len(self.bank_presets) - 1))
            self.bank_combo.setCurrentIndex(selected)
        else:
            selected = -1
            self.config["selected_bank_index"] = 0
        self.bank_combo.blockSignals(False)
        if selected >= 0:
            self.apply_bank_preset(selected)

    def apply_bank_preset(self, index):
        if 0 <= index < len(self.bank_presets):
            preset = self.bank_presets[index]
            self.spin_wait.setValue(float(preset.get("time", 20.5)))
            pilking_type = preset.get("pilking_type", "light")
            self.combo_pilking.setCurrentIndex(0 if pilking_type == "light" else 1)
            self.config["selected_bank_index"] = index

    def on_bank_selected(self, index):
        if index < 0 or index >= len(self.bank_presets):
            return
        self.apply_bank_preset(index)
        self.save_bank_presets()

    def save_bank_presets(self):
        self.config["bank_presets"] = self.bank_presets
        if self.bank_presets:
            self.config["selected_bank_index"] = max(0, min(int(self.config.get("selected_bank_index", 0)), len(self.bank_presets) - 1))
        else:
            self.config["selected_bank_index"] = 0
        path = os.path.join(CURRENT_DIR, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        if self.engine:
            self.engine.update_config(self.config)

    def add_bank(self):
        dialog = BankPresetDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        data = dialog.get_data()
        if not data["bank"] or not data["bait"]:
            QMessageBox.warning(self, "Не заполнено", "Укажи номер банки/ямы и название приманки.")
            return
        self.bank_presets.append(data)
        self.config["selected_bank_index"] = len(self.bank_presets) - 1
        self.refresh_bank_table()
        self.refresh_bank_combo()
        self.bank_table.selectRow(len(self.bank_presets) - 1)
        self.save_bank_presets()

    def edit_bank(self):
        row = self.bank_table.currentRow()
        if row < 0 or row >= len(self.bank_presets):
            return
        preset = self.bank_presets[row]
        dialog = BankPresetDialog(
            self,
            preset.get("bank", ""),
            preset.get("bait", ""),
            preset.get("time", 20.5),
            preset.get("pilking_type", "light")
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        data = dialog.get_data()
        if not data["bank"] or not data["bait"]:
            QMessageBox.warning(self, "Не заполнено", "Укажи номер банки/ямы и название приманки.")
            return
        self.bank_presets[row] = data
        self.config["selected_bank_index"] = row
        self.refresh_bank_table()
        self.refresh_bank_combo()
        self.bank_table.selectRow(row)
        self.save_bank_presets()

    def delete_bank(self):
        row = self.bank_table.currentRow()
        if row < 0 or row >= len(self.bank_presets):
            return
        answer = QMessageBox.question(self, "Удаление пресета", f'Удалить пресет банки «{self.bank_presets[row].get("bank", "")}»?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        self.bank_presets.pop(row)
        if self.bank_presets:
            self.config["selected_bank_index"] = min(row, len(self.bank_presets) - 1)
        else:
            self.config["selected_bank_index"] = 0
        self.refresh_bank_table()
        self.refresh_bank_combo()
        self.save_bank_presets()

    def create_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        group1 = QGroupBox("Пилкинг")
        g1 = QGridLayout(group1)
        g1.addWidget(QLabel("Лёгкий — удержание:"), 0, 0)
        self.spin_lh = QDoubleSpinBox()
        self.spin_lh.setRange(0.1, 5)
        self.spin_lh.setValue(0.3)
        self.spin_lh.setSuffix(" с")
        g1.addWidget(self.spin_lh, 0, 1)
        g1.addWidget(QLabel("Лёгкий — пауза:"), 1, 0)
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 10)
        self.spin_lw.setValue(2.1)
        self.spin_lw.setSuffix(" с")
        g1.addWidget(self.spin_lw, 1, 1)
        g1.addWidget(QLabel("Сильный — удержание:"), 2, 0)
        self.spin_hh = QDoubleSpinBox()
        self.spin_hh.setRange(0.1, 5)
        self.spin_hh.setValue(0.5)
        self.spin_hh.setSuffix(" с")
        g1.addWidget(self.spin_hh, 2, 1)
        g1.addWidget(QLabel("Сильный — пауза:"), 3, 0)
        self.spin_hw = QDoubleSpinBox()
        self.spin_hw.setRange(0.1, 10)
        self.spin_hw.setValue(2.3)
        self.spin_hw.setSuffix(" с")
        g1.addWidget(self.spin_hw, 3, 1)
        layout.addWidget(group1)
        
        group2 = QGroupBox("Цикл")
        g2 = QGridLayout(group2)
        g2.addWidget(QLabel("Задержка заброса:"), 0, 0)
        self.spin_cast = QDoubleSpinBox()
        self.spin_cast.setRange(0.5, 5)
        self.spin_cast.setValue(1.0)
        self.spin_cast.setSuffix(" с")
        g2.addWidget(self.spin_cast, 0, 1)
        g2.addWidget(QLabel("После поимки:"), 1, 0)
        self.spin_after = QDoubleSpinBox()
        self.spin_after.setRange(0.1, 3)
        self.spin_after.setValue(0.5)
        self.spin_after.setSuffix(" с")
        g2.addWidget(self.spin_after, 1, 1)
        layout.addWidget(group2)
        
        btn_save = QPushButton("💾 Сохранить")
        btn_save.clicked.connect(self.save_settings)
        layout.addWidget(btn_save)
        layout.addStretch()
        return tab
    
    def create_detection_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        monitor_group = QGroupBox("Монитор")
        mg = QGridLayout(monitor_group)
        mg.addWidget(QLabel("Выбор монитора:"), 0, 0)
        self.combo_monitor = QComboBox()
        mg.addWidget(self.combo_monitor, 0, 1)
        layout.addWidget(monitor_group)
        
        group = QGroupBox("Иконка поклёвки")
        g = QGridLayout(group)
        g.addWidget(QLabel("Файл:"), 0, 0)
        self.txt_icon = QLineEdit("bite_icon.png")
        g.addWidget(self.txt_icon, 0, 1)
        g.addWidget(QLabel("Порог:"), 1, 0)
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(0.5, 1)
        self.spin_thresh.setValue(0.8)
        g.addWidget(self.spin_thresh, 1, 1)
        g.addWidget(QLabel("X:"), 2, 0)
        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 4000)
        self.spin_x.setValue(804)
        g.addWidget(self.spin_x, 2, 1)
        g.addWidget(QLabel("Y:"), 3, 0)
        self.spin_y = QSpinBox()
        self.spin_y.setRange(0, 4000)
        self.spin_y.setValue(1318)
        g.addWidget(self.spin_y, 3, 1)
        g.addWidget(QLabel("Ширина:"), 4, 0)
        self.spin_w = QSpinBox()
        self.spin_w.setRange(10, 500)
        self.spin_w.setValue(133)
        g.addWidget(self.spin_w, 4, 1)
        g.addWidget(QLabel("Высота:"), 5, 0)
        self.spin_h = QSpinBox()
        self.spin_h.setRange(10, 500)
        self.spin_h.setValue(133)
        g.addWidget(self.spin_h, 5, 1)
        layout.addWidget(group)
        
        group2 = QGroupBox("Натяжение")
        g2 = QGridLayout(group2)
        g2.addWidget(QLabel("X:"), 0, 0)
        self.spin_tx = QSpinBox()
        self.spin_tx.setRange(0, 4000)
        self.spin_tx.setValue(900)
        g2.addWidget(self.spin_tx, 0, 1)
        g2.addWidget(QLabel("Y:"), 0, 2)
        self.spin_ty = QSpinBox()
        self.spin_ty.setRange(0, 4000)
        self.spin_ty.setValue(1412)
        g2.addWidget(self.spin_ty, 0, 3)
        layout.addWidget(group2)
        
        btn_test = QPushButton("🔍 Тест иконки")
        btn_test.clicked.connect(self.test_icon)
        layout.addWidget(btn_test)

        self.btn_live_score = QPushButton("📈 Старт мониторинга % совпадения")
        self.btn_live_score.setCheckable(True)
        self.btn_live_score.clicked.connect(self.toggle_live_score)
        layout.addWidget(self.btn_live_score)
        
        self.lbl_test = QLabel("")
        layout.addWidget(self.lbl_test)
        layout.addStretch()
        return tab
        
    def init_engine(self):
        self.engine = ClickerEngine(self.config)
        self.engine.bite_detected.connect(lambda: self.lbl_status.setText("🐟 ПОКЛЁВКА!"))
        self.engine.fish_caught.connect(lambda: self.lbl_status.setText("✅ Поймана!"))
        self.engine.status_changed.connect(self.statusBar.showMessage)
        self.engine.countdown_update.connect(lambda s: self.lbl_countdown.setText(str(s)))
        self.engine.hotkey_pressed.connect(self.on_hotkey)
        self.statusBar.showMessage(self.engine.hotkey_status, 10000)

        self.live_score_timer = QTimer()
        self.live_score_timer.timeout.connect(self.update_live_score)
        
    def toggle_action(self, key, enabled):
        self.action_states[key] = enabled
        self.engine.set_action_state(key, enabled)
        self.indicators[key].set_active(enabled)
        
    def toggle_auto(self, enabled):
        if enabled:
            self.config['auto_fishing'] = {
                'wait_after_cast': self.spin_wait.value(),
                'cast_delay': self.spin_cast.value(),
                'after_catch_delay': self.spin_after.value(),
                'pilking_type': 'light' if self.combo_pilking.currentIndex() == 0 else 'heavy'
            }
            self.engine.update_config(self.config)
            self.btn_start.setText("⏹ ОСТАНОВИТЬ")
            self.btn_start.setStyleSheet("background-color: #c62828;")
            self.engine.start_detection(self.chk_reel.isChecked(), self.chk_tension.isChecked())
        else:
            self.btn_start.setText("▶ ЗАПУСТИТЬ АВТО-ЛОВЛЮ")
            self.btn_start.setStyleSheet("")
            self.lbl_countdown.setText("")
            self.engine.stop_detection()
            
    def on_hotkey(self, action):
        new = not self.action_states[action]
        self.toggle_action(action, new)
        self.action_buttons[action].setChecked(new)
        
    def update_display(self):
        if self.engine:
            for k in self.action_states:
                st = self.engine.action_states.get(k, False)
                if self.action_states[k] != st:
                    self.action_states[k] = st
                    self.indicators[k].set_active(st)
                    self.action_buttons[k].setChecked(st)
                    
    def test_icon(self):
        r = self.engine.test_icon_detection(
            self.txt_icon.text(), self.spin_thresh.value(),
            (self.spin_x.value(), self.spin_y.value(), self.spin_w.value(), self.spin_h.value())
        )
        self.lbl_test.setText(r)

    def toggle_live_score(self, enabled):
        if enabled:
            self.btn_live_score.setText("⏹ Стоп мониторинга % совпадения")
            self.live_score_timer.start(400)
            self.update_live_score()
        else:
            self.btn_live_score.setText("📈 Старт мониторинга % совпадения")
            self.live_score_timer.stop()

    def update_live_score(self):
        try:
            score = self.engine.get_icon_match_score(
                self.txt_icon.text(),
                (self.spin_x.value(), self.spin_y.value(), self.spin_w.value(), self.spin_h.value())
            )
            if score is None:
                self.lbl_test.setText("❌ Не найден файл иконки")
                return
            self.lbl_test.setText(f"📈 Текущее совпадение: {score:.1%}")
        except Exception as e:
            self.lbl_test.setText(f"❌ Ошибка мониторинга: {str(e)}")

    def get_default_monitor_presets(self):
        return {
            'monitor_1': {
                'name': 'Монитор 1 (2560x1440)',
                'screen_resolution': {'width': 2560, 'height': 1440},
                'bite_icon': {'region': {'x': 804, 'y': 1318, 'width': 133, 'height': 133}},
                'tension_bar': {'check_x': 900, 'check_y': 1412}
            },
            'monitor_2': {
                'name': 'Монитор 2',
                'screen_resolution': {'width': 1920, 'height': 1080},
                'bite_icon': {'region': {'x': 550, 'y': 1024, 'width': 133, 'height': 133}},
                'tension_bar': {'check_x': 582, 'check_y': 1051}
            }
        }

    def ensure_monitor_presets(self):
        defaults = self.get_default_monitor_presets()
        presets = self.config.get('monitor_presets', {})

        if not presets:
            current_region = self.config.get('bite_icon', {}).get('region', {})
            current_tension = self.config.get('tension_bar', {})
            defaults['monitor_1']['bite_icon']['region'].update({
                'x': current_region.get('x', 804),
                'y': current_region.get('y', 1318),
                'width': current_region.get('width', 133),
                'height': current_region.get('height', 133)
            })
            defaults['monitor_1']['tension_bar'].update({
                'check_x': current_tension.get('check_x', 900),
                'check_y': current_tension.get('check_y', 1412)
            })
            self.config['monitor_presets'] = defaults
            presets = defaults
        else:
            for key, default_value in defaults.items():
                if key not in presets:
                    presets[key] = default_value
                else:
                    presets[key].setdefault('name', default_value['name'])
                    presets[key].setdefault('screen_resolution', default_value['screen_resolution'])
                    presets[key].setdefault('bite_icon', {})
                    presets[key]['bite_icon'].setdefault('region', default_value['bite_icon']['region'])
                    presets[key].setdefault('tension_bar', default_value['tension_bar'])

        self.monitor_presets = self.config['monitor_presets']
        self.config.setdefault('selected_monitor', 'monitor_1')

    def apply_monitor_preset(self, preset_key):
        preset = self.monitor_presets.get(preset_key)
        if not preset:
            return

        region = preset.get('bite_icon', {}).get('region', {})
        tension = preset.get('tension_bar', {})
        resolution = preset.get('screen_resolution', {})

        self.spin_x.setValue(region.get('x', 804))
        self.spin_y.setValue(region.get('y', 1318))
        self.spin_w.setValue(region.get('width', 133))
        self.spin_h.setValue(region.get('height', 133))
        self.spin_tx.setValue(tension.get('check_x', 900))
        self.spin_ty.setValue(tension.get('check_y', 1412))

        self.config['selected_monitor'] = preset_key
        self.config['screen_resolution'] = {
            'width': resolution.get('width', 2560),
            'height': resolution.get('height', 1440)
        }
        self.config.setdefault('bite_icon', {})
        self.config['bite_icon'].setdefault('enabled', True)
        self.config['bite_icon']['region'] = {
            'x': self.spin_x.value(),
            'y': self.spin_y.value(),
            'width': self.spin_w.value(),
            'height': self.spin_h.value()
        }
        self.config.setdefault('tension_bar', {})
        self.config['tension_bar']['check_x'] = self.spin_tx.value()
        self.config['tension_bar']['check_y'] = self.spin_ty.value()
        self.config['tension_bar'].setdefault('enabled', True)
        self.config['tension_bar'].setdefault('color_r_min', 180)

        if self.engine:
            self.engine.update_config(self.config)

    def on_monitor_changed(self):
        preset_key = self.combo_monitor.currentData()
        if preset_key:
            self.apply_monitor_preset(preset_key)
        
    def save_settings(self):
        selected_monitor = self.combo_monitor.currentData() or self.config.get('selected_monitor', 'monitor_1')

        self.config['presets'] = {
            'light_pilking': {'hold_time': self.spin_lh.value(), 'wait_time': self.spin_lw.value()},
            'heavy_pilking': {'hold_time': self.spin_hh.value(), 'wait_time': self.spin_hw.value()}
        }
        self.config['bite_icon'] = {
            'path': self.txt_icon.text(), 'threshold': self.spin_thresh.value(),
            'region': {'x': self.spin_x.value(), 'y': self.spin_y.value(), 'width': self.spin_w.value(), 'height': self.spin_h.value()}
        }
        self.config['tension_bar'] = {'check_x': self.spin_tx.value(), 'check_y': self.spin_ty.value(), 'color_r_min': 180}
        self.config['selected_monitor'] = selected_monitor
        self.config.setdefault('monitor_presets', self.get_default_monitor_presets())
        monitor_preset = self.config['monitor_presets'].setdefault(selected_monitor, {})
        monitor_preset.setdefault('name', selected_monitor)
        monitor_preset.setdefault('screen_resolution', self.config.get('screen_resolution', {'width': 2560, 'height': 1440}))
        monitor_preset['bite_icon'] = {'region': {'x': self.spin_x.value(), 'y': self.spin_y.value(), 'width': self.spin_w.value(), 'height': self.spin_h.value()}}
        monitor_preset['tension_bar'] = {'check_x': self.spin_tx.value(), 'check_y': self.spin_ty.value()}
        self.config['bank_presets'] = self.bank_presets
        self.config['auto_fishing'] = {
            'wait_after_cast': self.spin_wait.value(), 'cast_delay': self.spin_cast.value(),
            'after_catch_delay': self.spin_after.value(), 'pilking_type': 'light' if self.combo_pilking.currentIndex() == 0 else 'heavy'
        }
        path = os.path.join(CURRENT_DIR, 'config.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        self.engine.update_config(self.config)
        QMessageBox.information(self, "OK", "Сохранено!")
        
    def load_settings(self):
        p = self.config.get('presets', {})
        l = p.get('light_pilking', {})
        h = p.get('heavy_pilking', {})
        self.spin_lh.setValue(l.get('hold_time', 0.3))
        self.spin_lw.setValue(l.get('wait_time', 2.1))
        self.spin_hh.setValue(h.get('hold_time', 0.5))
        self.spin_hw.setValue(h.get('wait_time', 2.3))
        
        b = self.config.get('bite_icon', {})
        self.txt_icon.setText(b.get('path', 'bite_icon.png'))
        self.spin_thresh.setValue(b.get('threshold', 0.8))
        r = b.get('region', {})
        self.spin_x.setValue(r.get('x', 804))
        self.spin_y.setValue(r.get('y', 1318))
        self.spin_w.setValue(r.get('width', 133))
        self.spin_h.setValue(r.get('height', 133))
        
        t = self.config.get('tension_bar', {})
        self.spin_tx.setValue(t.get('check_x', 900))
        self.spin_ty.setValue(t.get('check_y', 1412))

        self.ensure_monitor_presets()
        selected_monitor = self.config.get('selected_monitor', 'monitor_1')
        self.combo_monitor.blockSignals(True)
        self.combo_monitor.clear()
        for key, preset in self.monitor_presets.items():
            self.combo_monitor.addItem(preset.get('name', key), key)
        selected_index = self.combo_monitor.findData(selected_monitor)
        if selected_index < 0:
            selected_index = 0
        self.combo_monitor.setCurrentIndex(selected_index)
        self.combo_monitor.blockSignals(False)
        self.combo_monitor.currentIndexChanged.connect(lambda _: self.on_monitor_changed())
        self.on_monitor_changed()
        
        f = self.config.get('auto_fishing', {})
        self.spin_wait.setValue(f.get('wait_after_cast', 32))
        self.bank_presets = self.config.get('bank_presets', [{'bank': '50', 'bait': 'Гигантский виброхвост 600гр', 'time': 20.5}])
        self.refresh_bank_table()
        self.refresh_bank_combo()
        self.spin_cast.setValue(f.get('cast_delay', 1.0))
        self.spin_after.setValue(f.get('after_catch_delay', 0.5))
        self.combo_pilking.setCurrentIndex(0 if f.get('pilking_type', 'light') == 'light' else 1)
        
    def closeEvent(self, event):
        if hasattr(self, 'live_score_timer'):
            self.live_score_timer.stop()
        if self.engine:
            self.engine.stop_all()
        event.accept()


# === ЗАПУСК ===
def load_config():
    path = os.path.join(CURRENT_DIR, 'config.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    config = load_config()
    window = MainWindow(config)
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()