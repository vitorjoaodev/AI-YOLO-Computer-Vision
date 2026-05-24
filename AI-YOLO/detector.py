"""
ObjectDetector - Módulo profissional de detecção de objetos
Suporte a YOLO-11N com PyTorch e OpenCV
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from ultralytics import YOLO
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ObjectDetector:
    """
    Classe profissional para detecção de objetos em imagens.
    Suporta múltiplos formatos e modelos YOLO.
    """
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}
    
    def __init__(
        self,
        model_name: str = "yolo11n",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Inicializa o detector de objetos.
        
        Args:
            model_name: Nome do modelo YOLO ('yolo11n', 'yolo11s', 'yolo11m', etc)
            confidence_threshold: Confiança mínima para detecções (0-1)
            iou_threshold: Threshold de IoU para NMS (0-1)
            device: Dispositivo ('cpu' ou 'cuda'). Auto-detecta se None
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Auto-detectar dispositivo
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Carregar modelo
        self._load_model()
    
    def _load_model(self):
        """Carrega o modelo YOLO com as configurações otimizadas."""
        try:
            logger.info(f"Carregando modelo {self.model_name}...")
            self.model = YOLO(f"{self.model_name}.pt")
            self.model.to(self.device)
            logger.info(f"Modelo {self.model_name} carregado com sucesso!")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
    
    def is_supported_format(self, image_path: str) -> bool:
        """Verifica se o formato da imagem é suportado."""
        return Path(image_path).suffix.lower() in self.SUPPORTED_FORMATS
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Carrega uma imagem com suporte a múltiplos formatos.
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Imagem como array numpy ou None se falhar
        """
        try:
            if not self.is_supported_format(image_path):
                logger.warning(f"Formato não suportado: {image_path}")
                return None
            
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Erro ao ler imagem: {image_path}")
                return None
            
            return image
        except Exception as e:
            logger.error(f"Erro ao carregar imagem: {e}")
            return None
    
    def detect(
        self,
        image: np.ndarray,
        return_crop: bool = False
    ) -> Dict:
        """
        Realiza detecção de objetos em uma imagem.
        
        Args:
            image: Imagem como array numpy (BGR)
            return_crop: Se True, retorna crops das detecções
            
        Returns:
            Dicionário com resultados da detecção
        """
        try:
            # Executar inferência
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            result = results[0]
            detections = []
            
            # Processar detecções
            if result.boxes is not None:
                boxes = result.boxes
                
                for idx, box in enumerate(boxes):
                    # Extrair coordenadas (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    detection_info = {
                        'id': idx,
                        'class': class_name,
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': {
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'width': x2 - x1,
                            'height': y2 - y1
                        }
                    }
                    
                    # Adicionar crop da detecção se solicitado
                    if return_crop:
                        crop = image[y1:y2, x1:x2]
                        detection_info['crop'] = crop
                    
                    detections.append(detection_info)
            
            return {
                'success': True,
                'num_detections': len(detections),
                'detections': detections,
                'image_shape': image.shape,
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Erro na detecção: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def detect_from_file(
        self,
        image_path: str,
        return_crop: bool = False
    ) -> Dict:
        """Detecção a partir de caminho de arquivo."""
        image = self.load_image(image_path)
        if image is None:
            return {'success': False, 'error': 'Falha ao carregar imagem'}
        
        return self.detect(image, return_crop=return_crop)
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        text_scale: float = 1.0,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Desenha as detecções na imagem.
        
        Args:
            image: Imagem original
            detections: Lista de detecções
            text_scale: Escala do texto
            thickness: Espessura das linhas
            
        Returns:
            Imagem com detecções desenhadas
        """
        output_image = image.copy()
        
        # Cores para diferentes classes (BGR)
        colors = {
            i: tuple(np.random.randint(0, 255, 3).tolist())
            for i in range(len(set(d['class_id'] for d in detections)))
        }
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            confidence = detection['confidence']
            class_name = detection['class']
            class_id = detection['class_id']
            
            # Desenhar bounding box
            color = colors.get(class_id, (0, 255, 0))
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            
            # Desenhar label
            label = f"{class_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)[0]
            label_y = max(y1, label_size[1] + 10)
            
            # Fundo do texto
            cv2.rectangle(
                output_image,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0] + 5, label_y + 5),
                color,
                -1
            )
            
            # Texto
            cv2.putText(
                output_image,
                label,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                (255, 255, 255),
                thickness
            )
        
        return output_image
    
    def save_result(
        self,
        image: np.ndarray,
        output_path: str,
        quality: int = 95
    ) -> bool:
        """
        Salva a imagem processada.
        
        Args:
            image: Imagem para salvar
            output_path: Caminho de saída
            quality: Qualidade JPEG (1-100)
            
        Returns:
            True se bem-sucedido
        """
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(output_path, image)
            
            logger.info(f"Imagem salva: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar imagem: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Retorna informações do modelo carregado."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'num_classes': len(self.model.names),
            'classes': self.model.names
        }
