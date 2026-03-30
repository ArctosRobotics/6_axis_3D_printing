"""
Bambu Lab Configuration Panel
UI for configuring and monitoring Bambu Lab printer connection

This panel provides a user interface for connecting to Bambu Lab 3D printers,
monitoring print status, and configuring automated workflows.

Author: Arctos Studio
Date: December 2024
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QProgressBar, QComboBox, QCheckBox, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QFont
import qtawesome as qta
from bambu_handler import BambuHandler, BambuPrinterStatus


class BambuPanel(QWidget):
    """Panel for Bambu Lab 3D printer configuration and monitoring"""
    
    def __init__(self, parent=None, bambu_handler=None):
        super().__init__(parent)
        self.parent_ui = parent
        self.bambu_handler = bambu_handler or BambuHandler(self)
        
        # Connect signals
        self.bambu_handler.connection_changed.connect(self.on_connection_changed)
        self.bambu_handler.status_changed.connect(self.on_status_changed)
        self.bambu_handler.print_progress.connect(self.on_progress_changed)
        self.bambu_handler.print_complete.connect(self.on_print_complete)
        self.bambu_handler.print_failed.connect(self.on_print_failed)
        self.bambu_handler.error_occurred.connect(self.on_error)
        
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Title
        title_label = QLabel("Bambu Lab 3D Printer")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)
        
        # Connection Settings Group
        connection_group = QGroupBox("Connection Settings")
        connection_layout = QVBoxLayout()
        
        # IP Address
        ip_layout = QHBoxLayout()
        ip_layout.addWidget(QLabel("IP Address:"))
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.100")
        self.ip_input.setToolTip("Printer IP address on local network")
        ip_layout.addWidget(self.ip_input)
        connection_layout.addLayout(ip_layout)
        
        # Serial Number
        serial_layout = QHBoxLayout()
        serial_layout.addWidget(QLabel("Serial Number:"))
        self.serial_input = QLineEdit()
        self.serial_input.setPlaceholderText("AC12309BH109")
        self.serial_input.setToolTip("Printer serial number (found in printer settings)")
        serial_layout.addWidget(self.serial_input)
        connection_layout.addLayout(serial_layout)
        
        # Access Code
        access_layout = QHBoxLayout()
        access_layout.addWidget(QLabel("Access Code:"))
        self.access_input = QLineEdit()
        self.access_input.setPlaceholderText("12347890")
        self.access_input.setEchoMode(QLineEdit.Password)
        self.access_input.setToolTip("Access code from printer settings (LAN Mode)")
        access_layout.addWidget(self.access_input)
        
        # Show/Hide password button
        self.show_password_btn = QPushButton()
        self.show_password_btn.setIcon(qta.icon('fa5s.eye'))
        self.show_password_btn.setFixedSize(30, 30)
        self.show_password_btn.setCheckable(True)
        self.show_password_btn.toggled.connect(self.toggle_password_visibility)
        self.show_password_btn.setToolTip("Show/Hide access code")
        access_layout.addWidget(self.show_password_btn)
        connection_layout.addLayout(access_layout)
        
        # Connect/Disconnect buttons
        button_layout = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setIcon(qta.icon('fa5s.plug'))
        self.connect_btn.clicked.connect(self.connect_to_printer)
        self.connect_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        button_layout.addWidget(self.connect_btn)
        
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setIcon(qta.icon('fa5s.times'))
        self.disconnect_btn.clicked.connect(self.disconnect_from_printer)
        self.disconnect_btn.setEnabled(False)
        self.disconnect_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        button_layout.addWidget(self.disconnect_btn)
        
        self.test_btn = QPushButton("Test")
        self.test_btn.setIcon(qta.icon('fa5s.vial'))
        self.test_btn.clicked.connect(self.test_connection)
        self.test_btn.setToolTip("Test connection and request status")
        button_layout.addWidget(self.test_btn)
        
        connection_layout.addLayout(button_layout)
        
        connection_group.setLayout(connection_layout)
        main_layout.addWidget(connection_group)
        
        # Status Display Group
        status_group = QGroupBox("Printer Status")
        status_layout = QVBoxLayout()
        
        # Connection status indicator
        conn_status_layout = QHBoxLayout()
        conn_status_layout.addWidget(QLabel("Connection:"))
        self.connection_indicator = QLabel("●")
        self.connection_indicator.setStyleSheet("color: #f44336; font-size: 20px;")
        self.connection_indicator.setToolTip("Disconnected")
        conn_status_layout.addWidget(self.connection_indicator)
        self.connection_label = QLabel("Disconnected")
        conn_status_layout.addWidget(self.connection_label)
        conn_status_layout.addStretch()
        status_layout.addLayout(conn_status_layout)
        
        # Printer status
        printer_status_layout = QHBoxLayout()
        printer_status_layout.addWidget(QLabel("Status:"))
        self.status_label = QLabel("Unknown")
        self.status_label.setStyleSheet("font-weight: bold;")
        printer_status_layout.addWidget(self.status_label)
        printer_status_layout.addStretch()
        status_layout.addLayout(printer_status_layout)
        
        # Current file
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.file_label = QLabel("None")
        self.file_label.setWordWrap(True)
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        status_layout.addLayout(file_layout)
        
        # Progress bar
        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Print Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        status_layout.addLayout(progress_layout)
        
        # Layer info
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("Layer:"))
        self.layer_label = QLabel("0/0")
        layer_layout.addWidget(self.layer_label)
        layer_layout.addStretch()
        layer_layout.addWidget(QLabel("Remaining:"))
        self.time_label = QLabel("--:--")
        layer_layout.addWidget(self.time_label)
        status_layout.addLayout(layer_layout)
        
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)
        
        # Automation Group
        automation_group = QGroupBox("Automation")
        automation_layout = QVBoxLayout()
        
        # Enable automation checkbox
        self.auto_enable_checkbox = QCheckBox("Enable automatic program execution on print complete")
        self.auto_enable_checkbox.setToolTip("Automatically run a robot program when print finishes")
        automation_layout.addWidget(self.auto_enable_checkbox)
        
        # Program selection
        program_layout = QHBoxLayout()
        program_layout.addWidget(QLabel("Program:"))
        self.program_combo = QComboBox()
        self.program_combo.addItem("(Select Program)")
        self.program_combo.setToolTip("Robot program to execute when print completes")
        program_layout.addWidget(self.program_combo)
        
        self.refresh_programs_btn = QPushButton()
        self.refresh_programs_btn.setIcon(qta.icon('fa5s.sync'))
        self.refresh_programs_btn.setFixedSize(30, 30)
        self.refresh_programs_btn.clicked.connect(self.refresh_program_list)
        self.refresh_programs_btn.setToolTip("Refresh program list")
        program_layout.addWidget(self.refresh_programs_btn)
        automation_layout.addLayout(program_layout)
        
        # Test trigger button
        self.test_trigger_btn = QPushButton("Test Trigger")
        self.test_trigger_btn.setIcon(qta.icon('fa5s.play'))
        self.test_trigger_btn.clicked.connect(self.test_trigger)
        self.test_trigger_btn.setToolTip("Test automation by running selected program")
        automation_layout.addWidget(self.test_trigger_btn)
        
        automation_group.setLayout(automation_layout)
        main_layout.addWidget(automation_group)
        
        # Log/Messages
        log_group = QGroupBox("Messages")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("background-color: #f5f5f5; font-family: monospace;")
        log_layout.addWidget(self.log_text)
        
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setIcon(qta.icon('fa5s.eraser'))
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_layout.addWidget(clear_log_btn)
        
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        # Add stretch to push everything to top
        main_layout.addStretch()
        
        # Initial program list refresh
        self.refresh_program_list()
    
    def toggle_password_visibility(self, checked):
        """Toggle password visibility"""
        if checked:
            self.access_input.setEchoMode(QLineEdit.Normal)
            self.show_password_btn.setIcon(qta.icon('fa5s.eye-slash'))
        else:
            self.access_input.setEchoMode(QLineEdit.Password)
            self.show_password_btn.setIcon(qta.icon('fa5s.eye'))
    
    def connect_to_printer(self):
        """Connect to Bambu Lab printer"""
        ip = self.ip_input.text().strip()
        serial = self.serial_input.text().strip()
        access_code = self.access_input.text().strip()
        
        if not ip or not serial or not access_code:
            self.log_message("ERROR: Please fill in all connection fields", "error")
            return
        
        self.log_message(f"Connecting to {ip}...")
        self.connect_btn.setEnabled(False)
        
        # Attempt connection
        success = self.bambu_handler.connect(ip, serial, access_code)
        
        if success:
            self.save_settings()
        else:
            self.connect_btn.setEnabled(True)
    
    def disconnect_from_printer(self):
        """Disconnect from printer"""
        self.log_message("Disconnecting...")
        self.bambu_handler.disconnect()
    
    def test_connection(self):
        """Test connection and request status"""
        if not self.bambu_handler.connected:
            self.log_message("ERROR: Not connected to printer", "error")
            return
        
        self.log_message("Requesting status update...")
        self.bambu_handler._request_status()
    
    @pyqtSlot(bool)
    def on_connection_changed(self, connected):
        """Handle connection state change"""
        if connected:
            self.connection_indicator.setStyleSheet("color: #4CAF50; font-size: 20px;")
            self.connection_indicator.setToolTip("Connected")
            self.connection_label.setText("Connected")
            self.connect_btn.setEnabled(False)
            self.disconnect_btn.setEnabled(True)
            self.log_message("[OK] Connected successfully", "success")
        else:
            self.connection_indicator.setStyleSheet("color: #f44336; font-size: 20px;")
            self.connection_indicator.setToolTip("Disconnected")
            self.connection_label.setText("Disconnected")
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.log_message("Disconnected", "warning")
    
    @pyqtSlot(str)
    def on_status_changed(self, status):
        """Handle printer status change"""
        self.status_label.setText(status.upper())
        
        # Update status label color
        if status == BambuPrinterStatus.PRINTING:
            self.status_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        elif status == BambuPrinterStatus.FINISH:
            self.status_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        elif status == BambuPrinterStatus.FAILED:
            self.status_label.setStyleSheet("font-weight: bold; color: #f44336;")
        elif status == BambuPrinterStatus.IDLE:
            self.status_label.setStyleSheet("font-weight: bold; color: #666;")
        else:
            self.status_label.setStyleSheet("font-weight: bold;")
        
        self.log_message(f"Status: {status.upper()}")
        
        # Update file and layer info
        status_data = self.bambu_handler.get_status()
        self.file_label.setText(status_data.get('file', 'None'))
        self.layer_label.setText(status_data.get('layer', '0/0'))
    
    @pyqtSlot(int)
    def on_progress_changed(self, progress):
        """Handle print progress update"""
        self.progress_bar.setValue(progress)
        
        # Update remaining time
        remaining = self.bambu_handler.get_remaining_time()
        if remaining > 0:
            hours = remaining // 3600
            minutes = (remaining % 3600) // 60
            self.time_label.setText(f"{hours:02d}:{minutes:02d}")
        else:
            self.time_label.setText("--:--")
    
    @pyqtSlot()
    def on_print_complete(self):
        """Handle print completion"""
        self.log_message("[OK] Print completed successfully!", "success")
        
        # Trigger automation if enabled
        if self.auto_enable_checkbox.isChecked():
            program_name = self.program_combo.currentText()
            if program_name and program_name != "(Select Program)":
                self.log_message(f"Triggering automation: {program_name}")
                self.trigger_program(program_name)
    
    @pyqtSlot(str)
    def on_print_failed(self, reason):
        """Handle print failure"""
        self.log_message(f"✗ Print failed: {reason}", "error")
    
    @pyqtSlot(str)
    def on_error(self, error_msg):
        """Handle error"""
        self.log_message(f"ERROR: {error_msg}", "error")
    
    def log_message(self, message, msg_type="info"):
        """Add message to log"""
        # Color code based on type
        if msg_type == "error":
            color = "#f44336"
        elif msg_type == "success":
            color = "#4CAF50"
        elif msg_type == "warning":
            color = "#FF9800"
        else:
            color = "#333"
        
        self.log_text.append(f'<span style="color: {color};">{message}</span>')
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def refresh_program_list(self):
        """Refresh list of available robot programs"""
        self.program_combo.clear()
        self.program_combo.addItem("(Select Program)")
        
        # Get programs from robot programmer
        if self.parent_ui and hasattr(self.parent_ui, 'robot_programmer'):
            programmer = self.parent_ui.robot_programmer
            if hasattr(programmer, 'program') and 'robots' in programmer.program:
                # Add robot programs
                for robot_idx, targets in programmer.program['robots'].items():
                    if targets:
                        program_name = f"Robot {robot_idx} Program"
                        self.program_combo.addItem(program_name, robot_idx)
        
        self.log_message("Program list refreshed")
    
    def test_trigger(self):
        """Test automation trigger"""
        program_name = self.program_combo.currentText()
        if not program_name or program_name == "(Select Program)":
            self.log_message("ERROR: Please select a program first", "error")
            return
        
        self.log_message(f"Testing trigger for: {program_name}")
        self.trigger_program(program_name)
    
    def trigger_program(self, program_name):
        """Trigger robot program execution"""
        if not self.parent_ui or not hasattr(self.parent_ui, 'robot_programmer'):
            self.log_message("ERROR: Robot programmer not available", "error")
            return
        
        try:
            # Get robot index from combo data
            robot_idx = self.program_combo.currentData()
            
            if robot_idx is not None:
                # Run the program
                programmer = self.parent_ui.robot_programmer
                if hasattr(programmer, 'run_program_with_transitions'):
                    self.log_message(f"Starting program execution...")
                    programmer.run_program_with_transitions()
                else:
                    self.log_message("ERROR: Program execution method not found", "error")
            else:
                self.log_message("ERROR: Invalid program selection", "error")
                
        except Exception as e:
            self.log_message(f"ERROR: Failed to trigger program: {str(e)}", "error")
    
    def save_settings(self):
        """Save connection settings to preferences"""
        if self.parent_ui and hasattr(self.parent_ui, 'preferences'):
            prefs = self.parent_ui.preferences
            prefs.set_bambu_settings(
                self.ip_input.text(),
                self.serial_input.text(),
                self.access_input.text(),
                self.auto_enable_checkbox.isChecked(),
                self.program_combo.currentText()
            )
            self.log_message("Settings saved")
    
    def load_settings(self):
        """Load connection settings from preferences"""
        if self.parent_ui and hasattr(self.parent_ui, 'preferences'):
            prefs = self.parent_ui.preferences
            settings = prefs.get_bambu_settings()
            
            self.ip_input.setText(settings.get('ip', ''))
            self.serial_input.setText(settings.get('serial', ''))
            self.access_input.setText(settings.get('access_code', ''))
            self.auto_enable_checkbox.setChecked(settings.get('auto_trigger_enabled', False))
            
            # Set program selection
            program_name = settings.get('auto_trigger_program', '')
            if program_name:
                index = self.program_combo.findText(program_name)
                if index >= 0:
                    self.program_combo.setCurrentIndex(index)
    
    def update_theme(self, is_dark_mode):
        """Update panel theme"""
        if is_dark_mode:
            # Dark theme
            self.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #3c3c3c;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    color: #e0e0e0;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QLabel {
                    color: #e0e0e0;
                }
                QLineEdit {
                    background-color: #323232;
                    border: 1px solid #3c3c3c;
                    color: #ffffff;
                    padding: 5px;
                }
                QTextEdit {
                    background-color: #1e1e1e;
                    color: #e0e0e0;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("")
