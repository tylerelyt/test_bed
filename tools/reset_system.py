#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键重置脚本 - 清理系统数据
用于教学环境维护和实验重置
"""

import os
import shutil
import glob
from datetime import datetime

def reset_system():
    """重置系统数据"""
    print("🔄 开始重置系统...")
    print("=" * 50)
    
    # 要清理的文件和目录
    files_to_remove = [
        'models/ctr_model.pkl',           # CTR模型文件
        'index_data.json',         # 索引文件（可选）
    ]
    
    # 要清理的文件模式
    patterns_to_remove = [
        'data/ctr_data.csv',            # CTR数据CSV文件
        'data/ctr_data.json',           # CTR数据JSON文件
        '*.log',                   # 日志文件
    ]
    
    # 清理文件
    removed_files = []
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"✅ 删除文件: {file_path}")
            except Exception as e:
                print(f"❌ 删除文件失败 {file_path}: {e}")
    
    # 清理匹配模式的文件
    for pattern in patterns_to_remove:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                removed_files.append(file_path)
                print(f"✅ 删除文件: {file_path}")
            except Exception as e:
                print(f"❌ 删除文件失败 {file_path}: {e}")
    
    # 清理临时目录
    temp_dirs = ['__pycache__', '.pytest_cache']
    for dir_name in temp_dirs:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"✅ 删除目录: {dir_name}")
            except Exception as e:
                print(f"❌ 删除目录失败 {dir_name}: {e}")
    
    # 重置统计
    print("=" * 50)
    print(f"📊 重置完成: 删除了 {len(removed_files)} 个文件")
    
    if removed_files:
        print("\n🗑️  已删除的文件:")
        for file_path in removed_files:
            print(f"   - {file_path}")
    
    print("\n💡 系统已重置，可以重新开始实验！")
    print("   建议操作:")
    print("   1. 运行 python ui/ui_interface.py 启动界面")
    print("   2. 进行搜索和点击操作，收集CTR数据")
    print("   3. 训练CTR模型，观察特征权重")
    print("   4. 切换排序方式，体验不同效果")

def backup_data():
    """备份当前数据"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = f"backup_{timestamp}"
    
    print(f"💾 备份数据到: {backup_dir}")
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        # 备份文件
        files_to_backup = [
            'models/ctr_model.pkl',
            'index_data.json',
        ]
        
        patterns_to_backup = [
            'data/ctr_data.json',
        ]
        
        backed_up_files = []
        
        for file_path in files_to_backup:
            if os.path.exists(file_path):
                shutil.copy2(file_path, backup_dir)
                backed_up_files.append(file_path)
                print(f"✅ 备份文件: {file_path}")
        
        for pattern in patterns_to_backup:
            for file_path in glob.glob(pattern):
                shutil.copy2(file_path, backup_dir)
                backed_up_files.append(file_path)
                print(f"✅ 备份文件: {file_path}")
        
        print(f"📦 备份完成: {len(backed_up_files)} 个文件已备份到 {backup_dir}")
        
    except Exception as e:
        print(f"❌ 备份失败: {e}")

def show_status():
    """显示系统状态"""
    print("📊 系统状态:")
    print("=" * 30)
    
    # 检查文件状态
    status_files = [
        ('CTR模型', 'models/ctr_model.pkl'),
        ('索引文件', 'index_data.json'),
    ]
    
    for name, file_path in status_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {name}: {file_path} ({size} bytes)")
        else:
            print(f"❌ {name}: 不存在")
    
    # 检查CTR数据文件
    ctr_file = 'data/ctr_data.json'
    if os.path.exists(ctr_file):
        size = os.path.getsize(ctr_file)
        print(f"📈 CTR数据文件: {ctr_file} ({size} bytes)")
    else:
        print("📈 CTR数据文件: 无")

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'reset':
            reset_system()
        elif command == 'backup':
            backup_data()
        elif command == 'status':
            show_status()
        else:
            print("❌ 未知命令。可用命令: reset, backup, status")
    else:
        # 交互模式
        print("🔄 系统重置工具")
        print("=" * 30)
        print("1. 重置系统 (删除所有数据)")
        print("2. 备份数据")
        print("3. 查看状态")
        print("4. 退出")
        
        while True:
            try:
                choice = input("\n请选择操作 (1-4): ").strip()
                
                if choice == '1':
                    confirm = input("⚠️  确定要重置系统吗？这将删除所有CTR数据和模型！(y/N): ")
                    if confirm.lower() == 'y':
                        reset_system()
                    else:
                        print("❌ 操作已取消")
                elif choice == '2':
                    backup_data()
                elif choice == '3':
                    show_status()
                elif choice == '4':
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择，请输入 1-4")
                    
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 操作失败: {e}")

if __name__ == "__main__":
    main() 