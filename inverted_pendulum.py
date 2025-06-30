import numpy as np
from numpy import sin, cos, arctan2
from itertools import cycle
from sys import argv, exit
import pyqtgraph as pg
from pyqtgraph import QtCore, QtWidgets, QtGui
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
from enum import Enum

class FuzzySet(Enum):
    """Enumeration for fuzzy set types"""
    LARGE_NEGATIVE = 0
    SMALL_NEGATIVE = 1
    ZERO = 2
    SMALL_POSITIVE = 3
    LARGE_POSITIVE = 4

@dataclass
class FuzzyConfig:
    """Configuration for fuzzy logic parameters"""
    theta_large: float = 30 * np.pi / 180
    dtheta_large: float = 30 * np.pi / 180
    x_large: float = 10
    dx_large: float = 15
    f_large: float = 250
    f_max: float = 500
    regulate_position: bool = False

class FuzzyLogic:
    """Fuzzy logic operations and membership functions"""
    
    @staticmethod
    def membership_base(x: float, a: float, b: float) -> float:
        """Base membership function: 0 at -inf, 1 at +inf"""
        if x < a:
            return 0
        elif x > b:
            return 1
        else:
            return (x - a) / (b - a)
    
    @staticmethod
    def membership_triangle(x: float, center: float, width: float) -> float:
        """Triangular membership function"""
        if x < center:
            return FuzzyLogic.membership_base(x, center - width/2, center)
        else:
            return 1 - FuzzyLogic.membership_base(x, center, center + width/2)
    
    @staticmethod
    def conjunction(values: np.ndarray) -> float:
        """T-norm (minimum)"""
        return np.min(values)
    
    @staticmethod
    def disjunction(values: np.ndarray) -> float:
        """T-conorm (maximum via De Morgan's law)"""
        return 1 - np.min(1 - values)

class MembershipFunctions:
    """Container for all membership functions"""
    
    def __init__(self, config: FuzzyConfig):
        self.config = config
    
    def _get_membership_values(self, x: float, large_value: float) -> dict:
        """Get all membership values for a given input"""
        return {
            FuzzySet.LARGE_NEGATIVE: 1 - FuzzyLogic.membership_base(x, -large_value, -large_value/2),
            FuzzySet.SMALL_NEGATIVE: FuzzyLogic.membership_triangle(x, -large_value/2, large_value),
            FuzzySet.ZERO: FuzzyLogic.membership_triangle(x, 0, large_value),
            FuzzySet.SMALL_POSITIVE: FuzzyLogic.membership_triangle(x, large_value/2, large_value),
            FuzzySet.LARGE_POSITIVE: FuzzyLogic.membership_base(x, large_value/2, large_value)
        }
    
    def theta_membership(self, theta: float) -> dict:
        return self._get_membership_values(theta, self.config.theta_large)
    
    def dtheta_membership(self, dtheta: float) -> dict:
        return self._get_membership_values(dtheta, self.config.dtheta_large)
    
    def x_membership(self, x: float) -> dict:
        return self._get_membership_values(x, self.config.x_large)
    
    def dx_membership(self, dx: float) -> dict:
        return self._get_membership_values(dx, self.config.dx_large)
    
    def force_membership(self, f: float) -> dict:
        return self._get_membership_values(f, self.config.f_large)

class FuzzyRules:
    """Fuzzy rule base for the controller"""
    
    def __init__(self, membership_funcs: MembershipFunctions):
        self.mf = membership_funcs
        self._define_rules()
    
    def _define_rules(self):
        """Define fuzzy rules for each output set"""
        # Rules for large negative force
        self.rules_large_negative = [
            (FuzzySet.LARGE_NEGATIVE, FuzzySet.LARGE_NEGATIVE),
            (FuzzySet.LARGE_NEGATIVE, FuzzySet.SMALL_NEGATIVE),
            (FuzzySet.SMALL_NEGATIVE, FuzzySet.LARGE_NEGATIVE),
            (FuzzySet.SMALL_NEGATIVE, FuzzySet.SMALL_NEGATIVE),
            (FuzzySet.LARGE_NEGATIVE, FuzzySet.ZERO),
            (FuzzySet.ZERO, FuzzySet.LARGE_NEGATIVE)
        ]
        
        # Rules for small negative force
        self.rules_small_negative = [
            (FuzzySet.ZERO, FuzzySet.SMALL_NEGATIVE),
            (FuzzySet.SMALL_NEGATIVE, FuzzySet.ZERO),
            (FuzzySet.SMALL_POSITIVE, FuzzySet.LARGE_NEGATIVE),
            (FuzzySet.LARGE_NEGATIVE, FuzzySet.SMALL_POSITIVE)
        ]
        
        # Rules for zero force
        self.rules_zero = [
            (FuzzySet.ZERO, FuzzySet.ZERO),
            (FuzzySet.SMALL_NEGATIVE, FuzzySet.SMALL_POSITIVE),
            (FuzzySet.LARGE_NEGATIVE, FuzzySet.LARGE_POSITIVE),
            (FuzzySet.SMALL_POSITIVE, FuzzySet.SMALL_NEGATIVE),
            (FuzzySet.LARGE_POSITIVE, FuzzySet.LARGE_NEGATIVE)
        ]
        
        # Rules for small positive force
        self.rules_small_positive = [
            (FuzzySet.SMALL_POSITIVE, FuzzySet.ZERO),
            (FuzzySet.ZERO, FuzzySet.SMALL_POSITIVE),
            (FuzzySet.LARGE_POSITIVE, FuzzySet.SMALL_NEGATIVE),
            (FuzzySet.SMALL_NEGATIVE, FuzzySet.LARGE_POSITIVE)
        ]
        
        # Rules for large positive force
        self.rules_large_positive = [
            (FuzzySet.LARGE_POSITIVE, FuzzySet.LARGE_POSITIVE),
            (FuzzySet.LARGE_POSITIVE, FuzzySet.SMALL_POSITIVE),
            (FuzzySet.SMALL_POSITIVE, FuzzySet.LARGE_POSITIVE),
            (FuzzySet.SMALL_POSITIVE, FuzzySet.SMALL_POSITIVE),
            (FuzzySet.LARGE_POSITIVE, FuzzySet.ZERO),
            (FuzzySet.ZERO, FuzzySet.LARGE_POSITIVE)
        ]
    
    def _evaluate_rules(self, rules: List, theta_mem: dict, dtheta_mem: dict, 
                       x_mem: dict = None, dx_mem: dict = None) -> float:
        """Evaluate a set of rules and return the maximum activation"""
        activations = []
        
        # Evaluate angle-based rules
        for theta_set, dtheta_set in rules:
            activation = FuzzyLogic.conjunction(np.array([
                theta_mem[theta_set],
                dtheta_mem[dtheta_set]
            ]))
            activations.append(activation)
        
        # Add position-based rules if position regulation is enabled
        if self.mf.config.regulate_position and x_mem is not None and dx_mem is not None:
            for x_set, dx_set in rules:
                activation = FuzzyLogic.conjunction(np.array([
                    x_mem[x_set],
                    dx_mem[dx_set]
                ]))
                activations.append(activation)
        
        return FuzzyLogic.disjunction(np.array(activations))
    
    def evaluate_all_rules(self, theta: float, dtheta: float, x: float, dx: float) -> np.ndarray:
        """Evaluate all rules and return coefficients for defuzzification"""
        theta_mem = self.mf.theta_membership(theta)
        dtheta_mem = self.mf.dtheta_membership(dtheta)
        x_mem = self.mf.x_membership(x)
        dx_mem = self.mf.dx_membership(dx)
        
        coefficients = np.array([
            self._evaluate_rules(self.rules_large_negative, theta_mem, dtheta_mem, x_mem, dx_mem),
            self._evaluate_rules(self.rules_small_negative, theta_mem, dtheta_mem, x_mem, dx_mem),
            self._evaluate_rules(self.rules_zero, theta_mem, dtheta_mem, x_mem, dx_mem),
            self._evaluate_rules(self.rules_small_positive, theta_mem, dtheta_mem, x_mem, dx_mem),
            self._evaluate_rules(self.rules_large_positive, theta_mem, dtheta_mem, x_mem, dx_mem)
        ])
        
        return coefficients

class Defuzzifier:
    """Defuzzification using center of gravity method"""
    
    def __init__(self, config: FuzzyConfig, membership_funcs: MembershipFunctions):
        self.config = config
        self.mf = membership_funcs
        self.force_domain = np.linspace(-config.f_max, config.f_max, 100)
    
    def defuzzify(self, coefficients: np.ndarray) -> float:
        """Defuzzify using center of gravity method"""
        # Create clipped membership functions
        clipped_memberships = np.zeros((5, len(self.force_domain)))
        force_memberships = [self.mf.force_membership(f) for f in self.force_domain]
        
        for i, force_set in enumerate(FuzzySet):
            for j, f in enumerate(self.force_domain):
                original_membership = force_memberships[j][force_set]
                clipped_memberships[i, j] = min(original_membership, coefficients[i])
        
        # Combine all membership functions (take maximum)
        combined_membership = np.max(clipped_memberships, axis=0)
        
        # Calculate center of gravity
        numerator = np.sum(combined_membership * self.force_domain)
        denominator = np.sum(combined_membership)
        
        return numerator / denominator if denominator != 0 else 0

class FuzzyController:
    """Main fuzzy controller class"""
    
    def __init__(self, config: FuzzyConfig = None):
        self.config = config or FuzzyConfig()
        self.membership_funcs = MembershipFunctions(self.config)
        self.rules = FuzzyRules(self.membership_funcs)
        self.defuzzifier = Defuzzifier(self.config, self.membership_funcs)
    
    def control(self, x: float, theta: float, dx: float, dtheta: float) -> float:
        """Compute control signal using fuzzy logic"""
        # Invert signs to match original convention
        theta = -theta
        dtheta = -dtheta
        x = -x
        
        # Evaluate rules
        coefficients = self.rules.evaluate_all_rules(theta, dtheta, x, dx)
        
        # Defuzzify
        return self.defuzzifier.defuzzify(coefficients)

class InvertedPendulum(QtWidgets.QWidget):
    """Inverted pendulum simulation with GUI"""
    
    def __init__(self, M=10, m=5, l=50, x0=0, theta0=0, dx0=0, dtheta0=0, 
                 dis_cyc=True, disruption=[0], iw=1000, ih=500, x_max=100, 
                 h_min=0, h_max=100, f_name=None):
        super().__init__()
        self.iter = 0
        
        # Load parameters from file or use defaults
        if f_name:
            self._load_from_file(f_name)
        else:
            self._set_parameters(M, m, l, x0, theta0, dx0, dtheta0, 
                               iw, ih, x_max, h_min, h_max, dis_cyc, disruption)
        
        # Initialize fuzzy controller
        config = FuzzyConfig(regulate_position=False)  # Set to True for position regulation
        self.fuzzy_controller = FuzzyController(config)
    
    def _load_from_file(self, f_name: str):
        """Load parameters from file"""
        with open(f_name) as f:
            lines = f.readlines()
            init_cond = lines[0].split(' ')
            self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = [
                float(el) for el in init_cond[:7]
            ]
            self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = [
                int(el) for el in init_cond[-5:]
            ]
            
            if lines[1].strip() == '1':
                self.disruption = cycle([float(el) for el in lines[2].split(' ')])
            else:
                self.disruption = iter([float(el) for el in lines[2].split(' ')])
    
    def _set_parameters(self, M, m, l, x0, theta0, dx0, dtheta0, iw, ih, x_max, h_min, h_max, dis_cyc, disruption):
        """Set parameters directly"""
        self.M, self.m, self.l = M, m, l
        self.x0, self.theta0, self.dx0, self.dtheta0 = x0, theta0, dx0, dtheta0
        self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = iw, ih, x_max, h_min, h_max
        
        if dis_cyc:
            self.disruption = cycle(disruption)
        else:
            self.disruption = iter(disruption)
    
    def init_image(self):
        """Initialize the GUI"""
        self.h_scale = self.image_h / (self.h_max - self.h_min)
        self.x_scale = self.image_w / (2 * self.x_max)
        self.hor = (self.h_max - 10) * self.h_scale
        self.c_w = 16 * self.x_scale
        self.c_h = 8 * self.h_scale
        self.r = 8
        
        # Initialize state
        self.x = self.x0
        self.theta = self.theta0
        self.dx = self.dx0
        self.dtheta = self.dtheta0
        
        self.setFixedSize(self.image_w, self.image_h)
        self.setWindowTitle("Inverted Pendulum")
        self.show()
        self.update()
    
    def paintEvent(self, e):
        """Draw the pendulum"""
        painter = QtGui.QPainter(self)
        
        # Ground line
        painter.setPen(pg.mkPen('k', width=2.0 * self.h_scale))
        painter.drawLine(0, int(self.hor), int(self.image_w), int(self.hor))
        
        # Pendulum rod
        painter.setPen(pg.mkPen((165, 42, 42), width=2.0 * self.x_scale))
        x_cart = self.x_scale * (self.x + self.x_max)
        x_bob = x_cart - self.x_scale * self.l * sin(self.theta)
        y_bob = self.hor - self.h_scale * self.l * cos(self.theta)
        painter.drawLine(int(x_cart), int(self.hor), int(x_bob), int(y_bob))
        
        # Cart
        painter.setPen(pg.mkPen('b'))
        painter.setBrush(pg.mkBrush('b'))
        painter.drawRect(int(x_cart - self.c_w/2), int(self.hor - self.c_h/2), 
                        int(self.c_w), int(self.c_h))
        
        # Pendulum bob
        painter.setPen(pg.mkPen('r'))
        painter.setBrush(pg.mkBrush('r'))
        painter.drawEllipse(int(x_bob - self.r * self.x_scale/2), 
                           int(y_bob - self.r * self.h_scale/2), 
                           int(self.r * self.x_scale), int(self.r * self.h_scale))
        
        # Scale markers
        painter.setPen(pg.mkPen('k'))
        for i in np.arange(-self.x_max, self.x_max, self.x_max/10):
            painter.drawText(int((i + self.x_max) * self.x_scale), 
                           int(self.image_h - 10), str(int(i)))
        
        for i in np.arange(self.h_min, self.h_max, (self.h_max - self.h_min)/10):
            painter.drawText(0, int(self.image_h - (int(i) - self.h_min) * self.h_scale), 
                           str(int(i)))
    
    def solve_equation(self, F: float) -> tuple:
        """Solve pendulum dynamics equations"""
        g = 9.81
        sin_theta, cos_theta = sin(self.theta), cos(self.theta)
        
        # System matrix
        a11 = self.M + self.m
        a12 = -self.m * self.l * cos_theta
        a21 = -cos_theta
        a22 = self.l
        
        # Right-hand side
        b1 = F - self.m * self.l * self.dtheta**2 * sin_theta
        b2 = g * sin_theta
        
        # Solve linear system
        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        solution = np.linalg.solve(A, b)
        
        return solution[0], solution[1]  # ddx, ddtheta
    
    def update_state(self, F: float, dt: float = 0.001):
        """Update system state using numerical integration"""
        ddx, ddtheta = self.solve_equation(F)
        
        # Integrate accelerations
        self.dx += ddx * dt
        self.x += self.dx * dt
        self.dtheta += ddtheta * dt
        self.theta += self.dtheta * dt
        
        # Normalize angle to [-π, π]
        self.theta = arctan2(sin(self.theta), cos(self.theta))
    
    def run(self, sandbox: bool = True, frameskip: int = 20):
        """Start the simulation"""
        self.sandbox = sandbox
        self.frameskip = frameskip
        self.init_image()
        
        # Setup timer for animation
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.simulation_step)
        timer.start(1)
        
        # Initialize plot
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Control Signal')
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Force')
    
    def simulation_step(self):
        """Single simulation step"""
        for _ in range(self.frameskip + 1):
            # Get disturbance
            disturbance = next(self.disruption, 0)
            
            # Compute control signal
            control = self.fuzzy_controller.control(self.x, self.theta, self.dx, self.dtheta)
            
            # Apply forces
            total_force = disturbance + control
            self.update_state(total_force)
            
            # Check bounds in non-sandbox mode
            if not self.sandbox:
                if (abs(self.x) > self.x_max or abs(self.theta) > np.pi/3):
                    exit(1)
        
        # Update visualization
        self.update()
        
        # Update plot
        self.ax.scatter(self.iter, control, c='blue', s=1)
        plt.pause(0.001)
        self.iter += 1

if __name__ == '__main__':
    app = QtWidgets.QApplication(argv)
    
    if len(argv) > 1:
        pendulum = InvertedPendulum(f_name=argv[1])
    else:
        pendulum = InvertedPendulum(
            x0=0, dx0=0, theta0=np.pi/10, dtheta0=0, 
            ih=800, iw=1000, h_min=-80, h_max=80
        )
    
    pendulum.run(sandbox=True)
    exit(app.exec_())