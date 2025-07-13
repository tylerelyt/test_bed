#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务管理器 - 统一管理所有服务实例
解决多层依赖传递问题，提供单一服务入口
"""

from typing import Optional
from .data_service import DataService
from .index_service import IndexService
from .model_service import ModelService


class ServiceManager:
    """服务管理器 - 单例模式管理所有服务"""
    
    _instance: Optional['ServiceManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._data_service: Optional[DataService] = None
            self._index_service: Optional[IndexService] = None
            self._model_service: Optional[ModelService] = None
            self._initialized = True
    
    @property
    def data_service(self) -> DataService:
        """获取数据服务实例"""
        if self._data_service is None:
            print("🚀 初始化数据服务...")
            self._data_service = DataService()
        return self._data_service
    
    @property
    def index_service(self) -> IndexService:
        """获取索引服务实例"""
        if self._index_service is None:
            print("🚀 初始化索引服务...")
            self._index_service = IndexService()
        return self._index_service
    
    @property
    def model_service(self) -> ModelService:
        """获取模型服务实例"""
        if self._model_service is None:
            print("🚀 初始化模型服务...")
            self._model_service = ModelService()
        return self._model_service
    
    def get_service_status(self) -> dict:
        """获取所有服务状态"""
        return {
            'data_service': {
                'status': 'running' if self._data_service else 'not_initialized',
                'samples_count': len(self.data_service.get_all_samples()) if self._data_service else 0
            },
            'index_service': {
                'status': 'running' if self._index_service else 'not_initialized',
                'documents_count': self.index_service.get_stats()['total_documents'] if self._index_service else 0
            },
            'model_service': {
                'status': 'running' if self._model_service else 'not_initialized',
                'is_trained': self.model_service.get_model_info()['is_trained'] if self._model_service else False
            }
        }
    
    def reset_services(self):
        """重置所有服务"""
        self._data_service = None
        self._index_service = None
        self._model_service = None
        print("🔄 所有服务已重置")


# 全局服务管理器实例
service_manager = ServiceManager()


def get_data_service() -> DataService:
    """获取数据服务实例"""
    return service_manager.data_service


def get_index_service() -> IndexService:
    """获取索引服务实例"""
    return service_manager.index_service


def get_model_service() -> ModelService:
    """获取模型服务实例"""
    return service_manager.model_service 