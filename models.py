import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

class LearnableYModel(nn.Module):
    def __init__(self, sentence_model_name='sentence-transformers/all-mpnet-base-v2',sentence_embed_dim=768, secret_dim=10, hidden_dim=1024, output_dim=128,output_range_max=1):
        super(LearnableYModel, self).__init__()
        self.sentence_encoder = AutoModel.from_pretrained(sentence_model_name)
        self.output_range_max=output_range_max
        self.secret_mlp = nn.Sequential(
            nn.Linear(secret_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(sentence_embed_dim + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )

    def encode_sentence(self, sentence,attention_mask):
        with torch.no_grad():
            model_output = self.sentence_encoder(sentence,attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, sentence,attention_mask, secret):
        encoded_sentence = self.encode_sentence(sentence,attention_mask)
        processed_secret = self.secret_mlp(secret)
        
        combined_input = torch.cat((encoded_sentence, processed_secret), dim=-1)
        output = self.fusion_mlp(combined_input)*self.output_range_max
        
        return output,encoded_sentence



class SecretEncoderBert(nn.Module):
    def __init__(self, bert_model_name, llm_tokenizer, hidden_dim, device):
        super(SecretEncoderBert, self).__init__()
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.llm_tokenizer = llm_tokenizer
        self.device = device

        self.condition_head = nn.Linear(self.bert_model.config.hidden_size, hidden_dim)

    def forward(self, condition_y_tokens):
        condition_y_text = self.llm_tokenizer.decode(condition_y_tokens, skip_special_tokens=True)
        
        inputs = self.bert_tokenizer(condition_y_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        bert_outputs = self.bert_model(**inputs)
        
        condition_y = bert_outputs.last_hidden_state[:, 0, :]
        condition_y = self.condition_head(condition_y)
        
        return condition_y

    
class TransformerGFWithSecret(nn.Module):
    def __init__(self, input_dim, output_dim,secret_dim, num_layers=2, num_heads=8, dim_feedforward=1024,dropout=0.1, hidden_dim=1024):
        super(TransformerGFWithSecret, self).__init__()
        self.secret_encoder = nn.Sequential(
            nn.Linear(secret_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,hidden_dim)
        )
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,dropout=dropout)
        self.transformer_g = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.f = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512,output_dim),
            nn.Tanh()
        )
            
    def forward(self, x, secret, attention_mask=None):
        batch_size=x.shape[0]
        x = self.input_encoder(x)
        secret = self.secret_encoder(secret).unsqueeze(1)
        x = torch.cat([secret, x], dim=1)

        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            # Create padding for secret token using x's device instead of src_key_padding_mask's device
            src_key_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device), src_key_padding_mask], dim=1)
        else:
            # If no attention mask, create a mask of all zeros with the correct device
            src_key_padding_mask = torch.zeros((batch_size, x.size(1) + 1), dtype=torch.bool, device=x.device)
        
        transformer_output = self.transformer_g(x.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        output = transformer_output[0, :, :]
        return self.f(output)

    def forward_train(self, x, secret, attention_mask=None):
        batch_size=x.shape[0]
        x = self.input_encoder(x)
        secret_encoded = self.secret_encoder(secret).unsqueeze(1)
        x = torch.cat([secret_encoded, x], dim=1)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
            src_key_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=src_key_padding_mask.device), src_key_padding_mask], dim=1)

        transformer_output = self.transformer_g(x.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        output = transformer_output[0, :, :]
        output = self.f(output)
        return output,secret_encoded

    def forward_no_f(self, x, secret, attention_mask=None):
        batch_size = x.shape[0]
        x = self.input_encoder(x)
        secret = self.secret_encoder(secret).unsqueeze(1)
        x = torch.cat([secret, x], dim=1)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
            src_key_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=src_key_padding_mask.device), src_key_padding_mask], dim=1)

        transformer_output = self.transformer_g(x.transpose(0, 1), src_key_padding_mask=src_key_padding_mask)
        output = transformer_output[0, :, :]
        return output

class MLPGFWithSecret(nn.Module):
    def __init__(self, input_dim, output_dim, secret_dim, hidden_dim=1024, num_layers=4,dropout_rate=0.1):
        super(MLPGFWithSecret, self).__init__()
        self.secret_encoder = nn.Sequential(
            nn.Linear(secret_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.residual_layers.append(nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x, secret):
        secret_embedding = self.secret_encoder(secret)
        x_embedding = self.input_mlp(x)

        combined_embedding = torch.cat([x_embedding, secret_embedding], dim=-1) 

        for layer in self.residual_layers:
            combined_embedding = layer(combined_embedding) + combined_embedding

        output = self.output_layer(combined_embedding)

        return output


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=2, num_heads=8, dim_feedforward=1024, dropout=0.1, hidden_dim=1024):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        encoder_layer = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward,dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
            
    def forward(self, x, attention_mask=None):
        x = self.encoder(x)
        batch_size = x.size(0)
        embedded = self.embedding.expand(batch_size, -1, -1)
        x = torch.cat([embedded, x], dim=1)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
            src_key_padding_mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=src_key_padding_mask.device), src_key_padding_mask], dim=1)

        transformer_output = self.transformer_encoder(x.transpose(0,1), src_key_padding_mask=src_key_padding_mask)
        cls_output = transformer_output[0, :, :]
        return self.mlp(cls_output)



class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets, mask=None):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        else:
            F_loss = (1 - pt) ** self.gamma * BCE_loss

        if mask is not None:
            F_loss = F_loss * mask

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def distance_loss(cls_output_1, cls_output_2, margin=1.0):
    distance = F.pairwise_distance(cls_output_1, cls_output_2, p=2)
    loss = F.relu(margin - distance)
    return loss.mean()

def combined_loss(outputs_1,outputs_2,cls_output_1,cls_output_2,labels,criterion,margin=0.1,gamma=0.5):
    cls_loss_1 = criterion(outputs_1,labels)
    cls_loss_2 = criterion(outputs_2,labels)
    dist_loss = distance_loss(cls_output_1,cls_output_2,margin)
    loss = cls_loss_1 + cls_loss_2 + gamma * dist_loss
    return loss

def mask_hidden_states(hs):
    result = hs.clone()
    result[:,::2,:]=0.
    return result
