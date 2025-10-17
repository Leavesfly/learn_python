import random
import string
import re
import hashlib
import json
import os
from datetime import datetime
import base64

class PasswordGenerator:
    def __init__(self):
        self.lowercase = string.ascii_lowercase
        self.uppercase = string.ascii_uppercase
        self.digits = string.digits
        self.symbols = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # 常见弱密码列表
        self.common_passwords = {
            '123456', 'password', '123456789', '12345678', '12345',
            '1234567', '1234567890', 'qwerty', 'abc123', 'million2',
            '000000', '1234', 'iloveyou', 'aaron431', 'password1',
            'qqww1122', '123', 'omgpop', '123321', '654321'
        }

    def generate_password(self, length=12, use_uppercase=True, use_lowercase=True, 
                         use_digits=True, use_symbols=True, exclude_ambiguous=True):
        """生成密码"""
        if length < 4:
            raise ValueError("密码长度至少为4位")
        
        char_pool = ""
        required_chars = []
        
        if use_lowercase:
            chars = self.lowercase
            if exclude_ambiguous:
                chars = chars.replace('l', '').replace('o', '')
            char_pool += chars
            required_chars.append(random.choice(chars))
            
        if use_uppercase:
            chars = self.uppercase
            if exclude_ambiguous:
                chars = chars.replace('I', '').replace('O', '')
            char_pool += chars
            required_chars.append(random.choice(chars))
            
        if use_digits:
            chars = self.digits
            if exclude_ambiguous:
                chars = chars.replace('0', '').replace('1', '')
            char_pool += chars
            required_chars.append(random.choice(chars))
            
        if use_symbols:
            char_pool += self.symbols
            required_chars.append(random.choice(self.symbols))
        
        if not char_pool:
            raise ValueError("至少需要选择一种字符类型")
        
        # 生成剩余字符
        remaining_length = length - len(required_chars)
        password_list = required_chars + [random.choice(char_pool) for _ in range(remaining_length)]
        
        # 随机打乱
        random.shuffle(password_list)
        return ''.join(password_list)

    def generate_memorable_password(self, word_count=4):
        """生成易记住的密码"""
        words = [
            'apple', 'brave', 'cloud', 'dance', 'eagle', 'flame', 'green', 'happy',
            'island', 'jungle', 'knight', 'light', 'magic', 'nature', 'ocean', 'peace',
            'quiet', 'river', 'storm', 'trust', 'unity', 'voice', 'water', 'youth'
        ]
        
        selected_words = random.sample(words, word_count)
        # 随机大写某些单词的首字母
        for i in range(len(selected_words)):
            if random.random() > 0.5:
                selected_words[i] = selected_words[i].capitalize()
        
        # 添加数字和符号
        number = random.randint(10, 99)
        symbol = random.choice('!@#$%')
        
        return ''.join(selected_words) + str(number) + symbol

class PasswordAnalyzer:
    def __init__(self):
        self.generator: PasswordGenerator = PasswordGenerator()

    def analyze_strength(self, password):
        """分析密码强度"""
        score = 0
        feedback = []
        
        # 长度检查
        if len(password) >= 12:
            score += 25
        elif len(password) >= 8:
            score += 15
            feedback.append("建议密码长度至少12位")
        else:
            score += 5
            feedback.append("密码长度过短，建议至少8位")
        
        # 字符类型检查
        has_lower: bool = bool(re.search(r'[a-z]', password))
        has_upper = bool(re.search(r'[A-Z]', password))
        has_digit = bool(re.search(r'\d', password))
        has_symbol = bool(re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', password))
        
        char_types = sum([has_lower, has_upper, has_digit, has_symbol])
        score += char_types * 15
        
        if char_types < 3:
            feedback.append("建议包含大小写字母、数字和特殊符号")
        
        # 重复字符检查
        repeated_chars = len(password) - len(set(password))
        if repeated_chars > len(password) * 0.3:
            score -= 10
            feedback.append("重复字符过多")
        
        # 连续字符检查
        consecutive_count = 0
        for i in range(len(password) - 2):
            if (ord(password[i+1]) == ord(password[i]) + 1 and 
                ord(password[i+2]) == ord(password[i]) + 2):
                consecutive_count += 1
        
        if consecutive_count > 0:
            score -= consecutive_count * 5
            feedback.append("避免使用连续字符")
        
        # 常见密码检查
        if password.lower() in self.generator.common_passwords:
            score = 0
            feedback.append("这是常见弱密码，强烈建议更换")
        
        # 字典单词检查
        common_words = ['password', 'admin', 'user', 'login', 'welcome']
        for word in common_words:
            if word in password.lower():
                score -= 15
                feedback.append(f"避免使用常见单词: {word}")
        
        score = max(0, min(100, score))
        
        if score >= 80:
            strength = "非常强"
            color = "🟢"
        elif score >= 60:
            strength = "强"
            color = "🟡"
        elif score >= 40:
            strength = "中等"
            color = "🟠"
        else:
            strength = "弱"
            color = "🔴"
        
        return {
            'score': score,
            'strength': strength,
            'color': color,
            'feedback': feedback,
            'details': {
                'length': len(password),
                'has_lower': has_lower,
                'has_upper': has_upper,
                'has_digit': has_digit,
                'has_symbol': has_symbol,
                'char_types': char_types
            }
        }

class PasswordManager:
    def __init__(self, master_password):
        self.master_password = master_password
        self.data_file = "passwords.dat"
        self.passwords = {}
        self.load_passwords()

    def _encrypt_data(self, data):
        """简单的数据加密"""
        key = hashlib.sha256(self.master_password.encode()).digest()
        data_json = json.dumps(data)
        data_bytes = data_json.encode('utf-8')
        
        # 简单XOR加密
        encrypted = bytes(a ^ b for a, b in zip(data_bytes, key * (len(data_bytes) // len(key) + 1)))
        return base64.b64encode(encrypted).decode('utf-8')

    def _decrypt_data(self, encrypted_data):
        """解密数据"""
        try:
            key = hashlib.sha256(self.master_password.encode()).digest()
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # 解密
            decrypted = bytes(a ^ b for a, b in zip(encrypted_bytes, key * (len(encrypted_bytes) // len(key) + 1)))
            data_json = decrypted.decode('utf-8')
            return json.loads(data_json)
        except:
            return {}

    def load_passwords(self):
        """加载密码数据"""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r') as f:
                    encrypted_data = f.read()
                    self.passwords = self._decrypt_data(encrypted_data)
            except:
                print("主密码错误或数据文件损坏")
                self.passwords = {}

    def save_passwords(self):
        """保存密码数据"""
        encrypted_data = self._encrypt_data(self.passwords)
        with open(self.data_file, 'w') as f:
            f.write(encrypted_data)

    def add_password(self, site, username, password, notes=""):
        """添加密码"""
        self.passwords[site] = {
            'username': username,
            'password': password,
            'notes': notes,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'last_modified': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_passwords()

    def get_password(self, site):
        """获取密码"""
        return self.passwords.get(site)

    def list_sites(self):
        """列出所有站点"""
        return list(self.passwords.keys())

    def delete_password(self, site):
        """删除密码"""
        if site in self.passwords:
            del self.passwords[site]
            self.save_passwords()
            return True
        return False

def main():
    generator = PasswordGenerator()
    analyzer = PasswordAnalyzer()
    
    print("🔐 智能密码生成器和管理器")
    print("=" * 50)
    
    while True:
        print("\n主菜单:")
        print("1. 生成随机密码")
        print("2. 生成易记密码")
        print("3. 密码强度分析")
        print("4. 密码管理器")
        print("5. 密码安全建议")
        print("6. 退出")
        
        choice = input("\n请选择功能 (1-6): ").strip()
        
        if choice == '1':
            print("\n🎲 密码生成设置:")
            try:
                length = int(input("密码长度 (默认12): ") or "12")
                use_uppercase = input("使用大写字母? (Y/n): ").lower() != 'n'
                use_lowercase = input("使用小写字母? (Y/n): ").lower() != 'n'
                use_digits = input("使用数字? (Y/n): ").lower() != 'n'
                use_symbols = input("使用特殊符号? (Y/n): ").lower() != 'n'
                exclude_ambiguous = input("排除易混淆字符? (Y/n): ").lower() != 'n'
                
                password = generator.generate_password(
                    length, use_uppercase, use_lowercase, 
                    use_digits, use_symbols, exclude_ambiguous
                )
                
                print(f"\n生成的密码: {password}")
                
                # 自动分析强度
                analysis = analyzer.analyze_strength(password)
                print(f"密码强度: {analysis['color']} {analysis['strength']} ({analysis['score']}/100)")
                
            except ValueError as e:
                print(f"错误: {e}")
        
        elif choice == '2':
            password = generator.generate_memorable_password()
            print(f"\n生成的易记密码: {password}")
            
            analysis = analyzer.analyze_strength(password)
            print(f"密码强度: {analysis['color']} {analysis['strength']} ({analysis['score']}/100)")
        
        elif choice == '3':
            password = input("\n请输入要分析的密码: ")
            if password:
                analysis = analyzer.analyze_strength(password)
                
                print(f"\n📊 密码分析结果:")
                print(f"强度: {analysis['color']} {analysis['strength']}")
                print(f"得分: {analysis['score']}/100")
                print(f"长度: {analysis['details']['length']} 字符")
                
                print("\n字符类型:")
                print(f"✓ 小写字母: {'是' if analysis['details']['has_lower'] else '否'}")
                print(f"✓ 大写字母: {'是' if analysis['details']['has_upper'] else '否'}")
                print(f"✓ 数字: {'是' if analysis['details']['has_digit'] else '否'}")
                print(f"✓ 特殊符号: {'是' if analysis['details']['has_symbol'] else '否'}")
                
                if analysis['feedback']:
                    print("\n💡 改进建议:")
                    for feedback in analysis['feedback']:
                        print(f"  • {feedback}")
        
        elif choice == '4':
            master_password = input("\n请输入主密码: ")
            if not master_password:
                print("主密码不能为空")
                continue
                
            manager = PasswordManager(master_password)
            
            while True:
                print("\n📋 密码管理器:")
                print("1. 添加密码")
                print("2. 查看密码")
                print("3. 列出所有站点")
                print("4. 删除密码")
                print("5. 返回主菜单")
                
                sub_choice = input("请选择操作: ").strip()
                
                if sub_choice == '1':
                    site = input("站点名称: ")
                    username = input("用户名: ")
                    password = input("密码 (留空自动生成): ")
                    
                    if not password:
                        password = generator.generate_password()
                        print(f"自动生成密码: {password}")
                    
                    notes = input("备注 (可选): ")
                    manager.add_password(site, username, password, notes)
                    print("密码已保存!")
                
                elif sub_choice == '2':
                    site = input("站点名称: ")
                    data = manager.get_password(site)
                    if data:
                        print(f"\n站点: {site}")
                        print(f"用户名: {data['username']}")
                        print(f"密码: {data['password']}")
                        print(f"备注: {data['notes']}")
                        print(f"创建时间: {data['created_at']}")
                    else:
                        print("未找到该站点的密码")
                
                elif sub_choice == '3':
                    sites = manager.list_sites()
                    if sites:
                        print("\n保存的站点:")
                        for i, site in enumerate(sites, 1):
                            print(f"{i}. {site}")
                    else:
                        print("暂无保存的密码")
                
                elif sub_choice == '4':
                    site = input("要删除的站点名称: ")
                    if manager.delete_password(site):
                        print("密码已删除!")
                    else:
                        print("未找到该站点")
                
                elif sub_choice == '5':
                    break
        
        elif choice == '5':
            print("\n🛡️ 密码安全建议:")
            print("1. 使用至少12位字符的密码")
            print("2. 包含大小写字母、数字和特殊符号")
            print("3. 避免使用个人信息(生日、姓名等)")
            print("4. 不要在多个网站使用相同密码")
            print("5. 定期更换重要账户密码")
            print("6. 启用双因素认证(2FA)")
            print("7. 使用密码管理器")
            print("8. 注意网络钓鱼攻击")
        
        elif choice == '6':
            print("感谢使用密码管理工具，再见! 🔒")
            break
        
        else:
            print("无效选择，请重新输入!")

if __name__ == "__main__":
    main()