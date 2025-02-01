from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Line, PushMatrix, Translate, PopMatrix, Rectangle
from kivy.core.window import Window
import os
from datetime import datetime
from kivy.core.text import LabelBase, Label as CoreLabel
from kivy.lang import Builder
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 定义神经网络模型
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(9216, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.conv2_drop(x)
        x = x.view(-1, 9216)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

class DrawingWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lines = []
        self.current_line = None
        # 设置固定的绘图区域大小
        self.drawing_size = 280  # 保持绘图区域为280x280
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        
        # 加载模型
        self.model = DigitCNN()
        try:
            self.model.load_state_dict(torch.load('digit_model.pth'))
            self.model.eval()
        except:
            print("未找到模型文件，请确保digit_model.pth存在")
        
        # 修改图像预处理流程
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # 改回28x28
            transforms.Grayscale(),       # 确保是灰度图
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def update_canvas(self, *args):
        # 计算绘图区域在窗口中的位置（居中）
        self.drawing_area = {
            'x': self.center_x - self.drawing_size/2,
            'y': self.center_y - self.drawing_size/2,
            'width': self.drawing_size,
            'height': self.drawing_size
        }
        # 清除并重绘边框
        self.canvas.before.clear()
        with self.canvas.before:
            Color(0.8, 0.8, 0.8)  # 灰色边框
            Line(rectangle=(self.drawing_area['x'], self.drawing_area['y'], 
                          self.drawing_area['width'], self.drawing_area['height']), width=2)

    def on_touch_down(self, touch):
        # 检查是否在绘图区域内
        if (self.drawing_area['x'] <= touch.x <= self.drawing_area['x'] + self.drawing_area['width'] and
            self.drawing_area['y'] <= touch.y <= self.drawing_area['y'] + self.drawing_area['height']):
            with self.canvas:
                Color(0, 1, 1)  # 青色线条
                self.current_line = Line(points=[touch.x, touch.y], width=10)
            return True

    def on_touch_move(self, touch):
        if self.current_line:
            # 限制绘制范围在区域内
            x = min(max(touch.x, self.drawing_area['x']), self.drawing_area['x'] + self.drawing_area['width'])
            y = min(max(touch.y, self.drawing_area['y']), self.drawing_area['y'] + self.drawing_area['height'])
            self.current_line.points += [x, y]
            return True

    def on_touch_up(self, touch):
        if self.current_line:
            self.lines.append(self.current_line)
            self.current_line = None

    def clear_canvas(self):
        # 保留边框，只清除绘图内容
        self.canvas.clear()
        self.lines = []
        self.update_canvas()

    def predict_digit(self):
        if not hasattr(self, 'model'):
            return "模型未加载"
        
        temp_file = "temp.png"
        
        # 创建一个新的Widget，大小设为绘图区域大小
        cropped_widget = Widget()
        cropped_widget.size = (self.drawing_size, self.drawing_size)
        
        with cropped_widget.canvas:
            PushMatrix()
            Translate(-self.drawing_area['x'], -self.drawing_area['y'])
            
            # 使用黑色背景
            Color(0, 0, 0)  # 黑色
            Rectangle(pos=(0, 0), size=(self.drawing_size, self.drawing_size))
            
            # 用白色绘制线条
            for line in self.lines:
                Color(1, 1, 1)  # 白色
                points = line.points
                Line(points=points, width=2)
            
            PopMatrix()
        
        cropped_widget.export_to_png(temp_file)
        
        # 打开图像并进行预处理
        image = Image.open(temp_file)
        image_tensor = self.transform(image).unsqueeze(0)
        
        # 进行预测
        with torch.no_grad():
            output = self.model(image_tensor)
            # 计算softmax获取概率分布
            probabilities = torch.exp(output).squeeze()
            # 转换为百分比列表
            probs = [f"{p.item()*100:.1f}%" for p in probabilities]
            # 获取最高概率的数字
            pred = output.argmax(dim=1, keepdim=True)
        
        os.remove(temp_file)
        # 返回概率分布和预测结果
        return pred.item(), probs

class DrawingApp(App):
    def build(self):
        # 设置默认字体为等线
        LabelBase.register(name='DengXian',
                          fn_regular='C:/Windows/Fonts/Deng.ttf')
        
        Builder.load_string('''
<Button>:
    font_name: 'DengXian'
<Label>:
    font_name: 'DengXian'
''')
        
        # 创建保存文件夹
        self.save_dir = 'drawings'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 创建主布局
        layout = BoxLayout(orientation='vertical')
        
        # 创建按钮布局
        button_layout = BoxLayout(size_hint_y=0.1)
        
        # 创建绘图区域
        self.drawing_widget = DrawingWidget()
        
        # 创建结果显示区域（包含标签和条形图）
        result_layout = BoxLayout(orientation='horizontal', size_hint_y=0.3)
        
        # 创建结果标签
        self.result_label = Label(
            text='绘制数字后点击识别', 
            size_hint_x=0.3,
            halign='left',
            valign='top'
        )
        self.result_label.bind(size=self.result_label.setter('text_size'))
        
        # 创建条形图Widget
        self.graph_widget = Widget(size_hint_x=0.7)
        
        # 添加到结果布局
        result_layout.add_widget(self.result_label)
        result_layout.add_widget(self.graph_widget)
        
        # 创建按钮
        clear_button = Button(text='清除')
        clear_button.bind(on_press=self.clear_drawing)
        
        save_button = Button(text='保存')
        save_button.bind(on_press=self.save_drawing)
        
        predict_button = Button(text='识别')
        predict_button.bind(on_press=self.predict_drawing)
        
        # 添加组件到布局
        button_layout.add_widget(clear_button)
        button_layout.add_widget(save_button)
        button_layout.add_widget(predict_button)
        
        layout.add_widget(self.drawing_widget)
        layout.add_widget(result_layout)
        layout.add_widget(button_layout)
        
        return layout

    def clear_drawing(self, instance):
        self.drawing_widget.clear_canvas()
        self.result_label.text = '绘制数字后点击识别'

    def save_drawing(self, instance):
        filename = f'drawing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        filepath = os.path.join(self.save_dir, filename)
        self.drawing_widget.export_to_png(filepath)

    def predict_drawing(self, instance):
        pred, probs = self.drawing_widget.predict_digit()
        # 更新文本结果
        self.result_label.text = f'预测结果：{pred}'
        
        # 更新条形图
        self.graph_widget.canvas.clear()
        with self.graph_widget.canvas:
            # 获取概率值（去掉百分号并转换为浮点数）
            values = [float(p.strip('%')) / 100 for p in probs]
            
            # 计算条形图的参数
            bar_spacing = 5
            bar_width = (self.graph_widget.width - bar_spacing * 11) / 10
            max_height = self.graph_widget.height * 0.8  # 留出更多空间给标签
            
            for i, value in enumerate(values):
                # 计算条形位置
                x = self.graph_widget.x + (bar_width + bar_spacing) * i
                height = max_height * value
                y = self.graph_widget.y + 25  # 为底部数字标签留出空间
                
                # 绘制条形
                if i == pred:
                    Color(0, 1, 0, 0.8)  # 预测结果用绿色
                else:
                    Color(0.5, 0.5, 0.5, 0.8)  # 其他用灰色
                Rectangle(pos=(x, y), size=(bar_width, height))
                
                # 添加底部数字标签
                Color(1, 1, 1, 1)  # 改为白色文字
                label = CoreLabel(text=str(i), font_size=14)
                label.refresh()
                texture = label.texture
                Rectangle(pos=(x + bar_width/2 - texture.width/2, y - 20),
                         size=texture.size, texture=texture)
                
                # 添加概率值标签（百分比）
                value_text = f'{value*100:.1f}%'
                label = CoreLabel(text=value_text, font_size=12)
                label.refresh()
                texture = label.texture
                
                # 所有标签都使用白色
                Color(1, 1, 1, 1)  # 白色文字
                label_y = y + height/2 - texture.height/2 if height > 30 else y + height + 5
                Rectangle(pos=(x + bar_width/2 - texture.width/2, label_y),
                         size=texture.size, texture=texture)

if __name__ == '__main__':
    Window.size = (800, 600)
    DrawingApp().run()
