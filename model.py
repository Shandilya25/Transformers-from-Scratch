import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    
    def __init__(self,vocab_size:int,embedding_dim:int):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        
    def forward(self,x):
        return self.embedding(x) * (math.sqrt(self.embedding_dim))
    
class PositionalEncoding(nn.Module):
    def __init__(self,embedding_size:int,seq_len:int,dropout:float)->None:
        '''Seq_len is to get the max sentence size which can be used for furture purpose such as padding of sentence etc.'''
        super().__init__()
        self.embedding_size=embedding_size
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)
        
        #We will create a matrix of embedding_dim x Sq_len. 
        # Imp:If it needs to move with the model (e.g., to GPU via .cuda()):
        #  Use PyTorch's register_buffer so it behaves correctly without being treated as a trainable parameter
        #and will be saved along with the state of the model
        matrix=torch.zeros([self.seq_len,self.embedding_size])
        
        pos_vector=torch.arange(0,self.seq_len,dtype=torch.float).unsqueeze(1)
        value_term=torch.exp(torch.arange(0,self.embedding_size,2).float() * (-math.log(10000.0)/self.embedding_size))
        matrix[:,0::2]=torch.sin(pos_vector*value_term)
        matrix[:,1::2]=torch.cos(pos_vector*value_term)
        
        matrix=matrix.unsqueeze(0)#(1,seq_len,embedding_dim)
        
        self.register_buffer('position_encoding',matrix)
        
    def forward(self,x):
        x=x+(self.position_encoding[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)
    

class LayerNormalization(nn.Module):
    
    def __init__(self,features,eps=1e-6):
        #eps is added to standard_dev under root==(root((sigma)^2+eps))
        super().__init__()
        self.eps=eps    
        self.gamma=nn.Parameter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features ))   
    
    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return self.gamma*(x-mean)/(std+self.eps) + self.beta


class ff_nn(nn.Module):
    def __init__(self,input_dim,output_dim,dropout):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.linear1=nn.Linear(input_dim,output_dim)
        self.dropout=nn.Dropout(dropout)
        self.linear2=nn.Linear(output_dim,input_dim)
    
    def forward(self,x):
        #inp: batch x seq_len x embedding_dim -> batch x seq_len x output_dim -> batch x seq_len x embedding_dim
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self,embedding_dim,head,dropout):
        super().__init__()
        self.embedding_dim=embedding_dim
        self.heads=head
        
        assert self.embedding_dim%self.heads==0,"Cannot divide the vector into given number of heads"
        
        self.embedding_div=self.embedding_dim//self.heads
        self.w_k=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_q=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.w_v=nn.Linear(self.embedding_dim,self.embedding_dim)
        
        self.final_w=nn.Linear(self.embedding_dim,self.embedding_dim)
        self.dropout=nn.Dropout(dropout)
        
    @staticmethod
    def find_attention(query,key,value,mask,dropout):
        
        d_k=query.shape[-1]
        attention_scores=(query @ key.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None:
            attention_scores=attention_scores.masked_fill(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
            
        if dropout is not None:
            attention_scores=dropout(attention_scores)
        return (attention_scores @ value),attention_scores

            
        
        
    def forward(self,q,k,v,mask):
        query=self.w_q(q) #-->Dimensions are(batch,Seq_len,embedding_dim) Same for both output and input
        key=self.w_k(k)   #-->Dimensions are(batch,Seq_len,embedding_dim)
        value=self.w_v(v) #-->Dimensions are(batch,Seq_len,embedding_dim)
        
        
        #Here the creation of these vectors are important because of the way we 
        # are converting or reshaping the given q,k,v vectors so that each vector into
        # a head of MH properly aligns with the inpput requirements and especially the usage 
        # of q.view(dims).transpose
        
        query=query.view(query.shape[0],query.shape[1],self.heads,self.embedding_div).transpose(1,2)
        key=key.view(key.shape[0],key.shape[1],self.heads,self.embedding_div).transpose(1,2)
        value=value.view(value.shape[0],value.shape[1],self.heads,self.embedding_div).transpose(1,2)
        
        x,self.attention_scores=MultiHeadAttention.find_attention(query,key,value,mask,self.dropout)
        # x=x.transpose(1,2)
        # x=x.view(x.shape[0],x.shape[1],self.head*self.embedding_div)
        
        ##The above method is wrong because operations like transpose,permute makes the 
        ## memory non contiguous because the strides are not looked-over so before making a 
        #  .view() method u have to use .contiguous methods
        # transpose() / permute(): Do not copy memory. They simply swap stride numbers. That is what causes non-contiguity.
        
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.heads*self.embedding_div)
        
        return self.final_w(x)


class ResidualConnection(nn.Module):
    def __init__(self,features,dropout):
        super().__init__() 
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization(features)
        
    def forward(self,x,prev_layer):
        return x+self.dropout(prev_layer(self.norm(x)))
        

        
class Encoder(nn.Module):
    def __init__(self,features,multi_head,FF,dropout):
        super().__init__()
        self.attention_block=multi_head
        self.feed_forward_block=FF
        self.residual_connection=nn.ModuleList([ResidualConnection(features,dropout) for _ in range(2)])
        
    def forward(self,x,src_mask):
        x=self.residual_connection[0](x,lambda x:self.attention_block(x,x,x,src_mask))
        x=self.residual_connection[1](x,self.feed_forward_block)
        return x
    
class FullEncoder(nn.Module):
    def __init__(self,features,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)
        
    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)
    
        
        
class Decoder(nn.Module):
    def __init__(self,features,multi_head,cross_attention,ff,dropout):
        super().__init__()
        self.mh_attention=multi_head
        self.cross_attention=cross_attention
        self.ff_block=ff
        self.residual_connection=nn.ModuleList([ResidualConnection(features,dropout) for _ in range(3)])
        
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        x=self.residual_connection[0](x,lambda x:self.mh_attention(x,x,x,tgt_mask))
        x=self.residual_connection[1](x,lambda x:self.cross_attention(x,encoder_output,encoder_output,src_mask))
        x=self.residual_connection[2](x,self.ff_block)
        
        return x 


class FullDecoder(nn.Module):
    def __init__(self,features,layers):
        super().__init__()
        self.layers=layers
        self.norm=LayerNormalization(features)
    
    def forward(self,x,encoder_output,src_mask,tgt_mask):
        for layer in self.layers:
            x=layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

class FinalLLayer(nn.Module):
    def __init__(self,decoder_output,output_dim):
        super().__init__()
        self.llayer=nn.Linear(decoder_output,output_dim)
        
    def forward(self,x):
        return torch.log_softmax(self.llayer(x),dim=-1)
    

    
class Transformer(nn.Module):
    def __init__(self,src_embedding,tgt_embedding,encoder,decoder,src_posn,tgt_posn,llayer):
        super().__init__()
        self.src_embedding=src_embedding
        self.tgt_embedding=tgt_embedding
        self.encoder=encoder 
        self.decoder=decoder
        self.src_posn=src_posn
        self.tgt_posn=tgt_posn
        self.llayer=llayer
    
    def encode(self,src,src_mask):
        src=self.src_embedding(src)
        src=self.src_posn(src)
        return self.encoder(src,src_mask)

    def decode(self,encoder_output,src_mask,tgt,tgt_mask):
        tgt=self.tgt_embedding(tgt)
        tgt=self.tgt_posn(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    def project(self, x):
        return self.llayer(x)
    

def build_transformer(src_vocab_size:int,tgt_vocab_size:int,src_seq_len:int,tgt_seq_len:int,
                      embedding_dim:int=512,N:int=6,heads:int=8,dropout:float=0.1,d_ff:int=2048)->Transformer:
    src_embedding=InputEmbedding(src_vocab_size,embedding_dim)
    tgt_embedding=InputEmbedding(tgt_vocab_size,embedding_dim)

    src_posn=PositionalEncoding(embedding_dim,src_seq_len,dropout)
    tgt_posn=PositionalEncoding(embedding_dim,tgt_seq_len,dropout)

    encoder_blocks=[]
    for _ in range(N):
        attention=MultiHeadAttention(embedding_dim,heads,dropout)
        ff=ff_nn(embedding_dim,d_ff,dropout)
        encoder_blocks.append(Encoder(embedding_dim,attention,ff,dropout))

    decoder_blocks=[]
    for _ in range(N):
        self_attention=MultiHeadAttention(embedding_dim,heads,dropout)
        cross_attention=MultiHeadAttention(embedding_dim,heads,dropout)
        ff=ff_nn(embedding_dim,d_ff,dropout)
        decoder_blocks.append(Decoder(embedding_dim,self_attention,cross_attention,ff,dropout))

    encoder=FullEncoder(embedding_dim,nn.ModuleList(encoder_blocks))
    decoder=FullDecoder(embedding_dim,nn.ModuleList(decoder_blocks))
    llayer=FinalLLayer(embedding_dim,tgt_vocab_size)

    transformer=Transformer(src_embedding,tgt_embedding,encoder,decoder,src_posn,tgt_posn,llayer)

    for p in transformer.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

    return transformer
