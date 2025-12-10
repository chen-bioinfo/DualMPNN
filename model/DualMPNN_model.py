from .DualMPNN_util import *

class DualMPNN(nn.Module):
    def __init__(self, 
                num_letters=21, 
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=21,
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()
        
        self.net_a = SingleStream(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout
        )
        
        self.net_b = SingleStream_P(
            num_letters=num_letters,
            node_features=node_features,
            edge_features=edge_features,
            hidden_dim=hidden_dim,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            vocab=vocab,
            k_neighbors=k_neighbors,
            augment_eps=augment_eps,
            dropout=dropout
        )
        
        self.cross_attn_layers = nn.ModuleList([
            AlignedCrossAttention(hidden_dim=hidden_dim)
            for _ in range(num_encoder_layers)
        ])

        self.cross_dec_attn_layers = nn.ModuleList([
            AlignedCrossAttention(hidden_dim=hidden_dim)
            for _ in range(num_decoder_layers)
        ])

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p,align_pairs,tms):
        # 'a' means Query Branch, 'b' means Template Branch
        B = X.size(0)
        E_a, E_idx_a = self.net_a.features(X, mask, residue_idx, chain_encoding_all)
        E_b, E_idx_b, h_V_b, h_E_b=[], [], [], []
        num_pairs = len(X_p)
        if num_pairs>0:
            num_pairs=1
        for k in range(num_pairs):
            eb, ei = self.net_b.features(X_p[k], mask_p[k], residue_idx_p[k], chain_encoding_all_p[k])
            E_b.append(eb)
            E_idx_b.append(ei)
        
      
        h_V_a, h_E_a = self.net_a.init_states(E_a, S)
        for k in range(num_pairs):
            hv, he = self.net_b.init_states_pair(E_b[k], S_p[k], self.net_a.W_s)
            h_V_b.append(hv)
            h_E_b.append(he)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx_a).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        
        aln_idx = []
        tm_score = []
            
        for k in range(num_pairs):
            tm_score.append([b[k] for b in tms]) # tms:[B,P,L]
            aln_idx.append([b[k] for b in align_pairs]) # align_pairs:[B,P,L_aln,2]

        for k in range(num_pairs):
            for i in range(B):
                align_pairs = aln_idx[k][i]
                a_indices = [p[0] for p in align_pairs]
                b_indices = [p[1] for p in align_pairs]
                h_V_a[i,a_indices,:] += h_V_b[k][i,b_indices,:] * tm_score[k][i]

        for (enc_a, enc_b), cross_attn in zip(zip(self.net_a.encoder_layers, self.net_b.encoder_layers), 
                                            self.cross_attn_layers):
            h_V_a, h_E_a = enc_a(h_V_a, h_E_a, E_idx_a, mask, mask_attend)
            for k in range(num_pairs):
                h_V_b[k], h_E_b[k] = enc_b(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        h_ES_b, h_EXV_enc_fw_b, mask_bw_b=[],[],[]
        h_ES_a, h_EXV_enc_fw_a, mask_bw_a = self.net_a._init_decoder(h_V_a, h_E_a, E_idx_a, mask, chain_M, S)
        for k in range(num_pairs):
            hesb, hec, mb = self.net_b._init_decoder(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k], chain_M_p[k], S_p[k], self.net_a.W_s)
            h_ES_b.append(hesb)
            h_EXV_enc_fw_b.append(hec)
            mask_bw_b.append(mb)

        for (dec_layer_a, dec_layer_b), cross_attn in zip(zip(self.net_a.decoder_layers, self.net_b.decoder_layers), 
                                                        self.cross_dec_attn_layers):
            h_V_a = self.net_a._decoder_step(h_V_a, dec_layer_a, h_ES_a, h_EXV_enc_fw_a, mask_bw_a, mask, E_idx_a)
            for k in range(num_pairs):
                h_V_b[k] = self.net_b._decoder_step(h_V_b[k], dec_layer_b, h_ES_b[k], h_EXV_enc_fw_b[k], mask_bw_b[k], mask_p[k], E_idx_b[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        logits = self.net_a.W_out(h_V_a)
        
        return F.log_softmax(logits, dim=-1)
    
    def sample(self, X, mask, chain_M, residue_idx, chain_encoding_all, X_p, S_p, mask_p, chain_M_p, residue_idx_p, chain_encoding_all_p,align_pairs,tms,temperature=1.0):

        B = X.size(0)
        E_a, E_idx_a = self.net_a.features(X, mask, residue_idx, chain_encoding_all)
        E_b, E_idx_b, h_V_b, h_E_b=[], [], [], []
        num_pairs = len(X_p)
        if num_pairs>0:
            num_pairs=1
        for k in range(num_pairs):
            eb, ei = self.net_b.features(X_p[k], mask_p[k], residue_idx_p[k], chain_encoding_all_p[k])
            E_b.append(eb)
            E_idx_b.append(ei)

        h_V_a, h_E_a = self.net_a.init_states(E_a)
        for k in range(num_pairs):
            hv, he = self.net_b.init_states_pair(E_b[k], S_p[k], self.net_a.W_s)
            h_V_b.append(hv)
            h_E_b.append(he)
        
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx_a).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend

        aln_idx = []
        tm_score = []
            
        for k in range(num_pairs):
            tm_score.append([b[k] for b in tms]) # tms:[B,P,L]
            aln_idx.append([b[k] for b in align_pairs]) # align_pairs:[B,P,L_aln,2]

        for k in range(num_pairs):
            for i in range(B):
                align_pairs = aln_idx[k][i]
                a_indices = [p[0] for p in align_pairs]
                b_indices = [p[1] for p in align_pairs]
                h_V_a[i,a_indices,:] += h_V_b[k][i,b_indices,:] * tm_score[k][i]
        # if num_pairs==0:
        #     h_V_a = self.net_a.W_s(torch.randint(low=0, high=21, size=(X.shape[0], X.shape[1]), device=X.device))

        for (enc_a, enc_b), cross_attn in zip(zip(self.net_a.encoder_layers, self.net_b.encoder_layers), 
                                            self.cross_attn_layers):

            h_V_a, h_E_a = enc_a(h_V_a, h_E_a, E_idx_a, mask, mask_attend)
            for k in range(num_pairs):
                h_V_b[k], h_E_b[k] = enc_b(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        h_ES_b, h_EXV_enc_fw_b, mask_bw_b=[],[],[]
        for k in range(num_pairs):
            hesb, hec, mb = self.net_b._init_decoder(h_V_b[k], h_E_b[k], E_idx_b[k], mask_p[k], chain_M_p[k], S_p[k], self.net_a.W_s)
            h_ES_b.append(hesb)
            h_EXV_enc_fw_b.append(hec)
            mask_bw_b.append(mb)

        for (dec_layer_a, dec_layer_b), cross_attn in zip(zip(self.net_a.decoder_layers, self.net_b.decoder_layers), 
                                                        self.cross_dec_attn_layers):

            h_V_a = self.net_a.sampler(h_V_a, h_E_a, E_idx_a, mask, chain_M, 0.01, dec_layer_a)
            for k in range(num_pairs):
                h_V_b[k] = self.net_b._decoder_step(h_V_b[k], dec_layer_b, h_ES_b[k], h_EXV_enc_fw_b[k], mask_bw_b[k], mask_p[k], E_idx_b[k])
                h_V_a, _ = cross_attn(h_V_a, h_V_b[k], mask, aln_idx[k], tm_score[k])

        logits = self.net_a.W_out(h_V_a) / 0.01
        probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, num_classes = probs.shape
        probs_2d = probs.reshape(-1, num_classes)  # (batch_size * seq_len, num_classes)
        samples_flat = torch.multinomial(probs_2d, 1)  # (batch_size * seq_len, 1)
        S = samples_flat.reshape(batch_size, seq_len)  # (batch_size, seq_len)
        return S
    

class SingleStream(nn.Module):
    def __init__(self, 
                num_letters=21,
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=21,
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, 
            edge_features, 
            top_k=k_neighbors, 
            augment_eps=augment_eps,
            CA_only = False
        )
        

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.encoder_layers = nn.ModuleList([
            EncLayer(
                hidden_dim, 
                hidden_dim*2,  
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(
                hidden_dim,
                hidden_dim*3,
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])
        
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_states(self, E, S=None):
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)
        return h_V, h_E

    def init_states_pair(self, E, S):
        h_V = initialize_h_V(self, E, S, onehot_dim=21, init_ratio=1.0, error_init_ratio=0.0)
        h_E = self.W_e(E)
        return h_V, h_E

    def decode(self, h_V, h_E, E_idx, mask, chain_M, S, cross_attn_layers=None, h_V_b=None, align_pairs=None):
        device = h_V.device
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for i, layer in enumerate(self.decoder_layers):
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
            if cross_attn_layers and h_V_b is not None:
                cross_attn = cross_attn_layers[i]
                h_V, _ = cross_attn(h_V, h_V_b, mask, align_pairs)
        
        return self.W_out(h_V)
    def sampler(self, h_V, h_E, E_idx, mask, chain_M, temperature, layer):
        # Decoder alternates masked self-attention
        device = h_V.device
        chain_mask = chain_M*mask #update chain_M to include missing regions
        decoding_order = torch.argsort((chain_mask+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        N_batch, N_nodes = h_V.size(0), h_V.size(1)
        log_probs = torch.zeros((N_batch, N_nodes, 20))
        h_S = torch.zeros_like(h_V)
        h_V_out = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64)
        # h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        for t in range(N_nodes):
            # Hidden layers
            E_idx_t = E_idx[:,t:t+1,:]
            h_E_t = h_E[:,t:t+1,:,:]
            h_ES_t = cat_neighbors_nodes(h_S, h_E_t, E_idx_t)
            # Stale relational features for future states
            h_ESV_encoder_t = mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
            # for l, layer in enumerate(self.decoder_layers):
                # Updated relational features for future states
            h_ESV_decoder_t = cat_neighbors_nodes(h_V, h_ES_t, E_idx_t)
            h_V_t = h_V[:,t:t+1,:]
            h_ESV_t = mask_bw[:,t:t+1,:,:] * h_ESV_decoder_t + h_ESV_encoder_t
            h_V[:,t,:] = layer(
                h_V_t, h_ESV_t, mask_V=mask[:,t:t+1]
            )
            logits = self.W_out(h_V_t) / temperature
        return h_V
    
    def _init_decoder(self, h_V, h_E, E_idx, mask, chain_M, S):
        device = h_V.device
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device))))
        
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        return h_ES, h_EXV_encoder_fw, mask_bw

    def _decoder_step(self, h_V, layer, h_ES, h_EXV_encoder_fw, mask_bw, mask, E_idx):
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        return layer(h_V, h_EXV_encoder_fw, mask)
    

class SingleStream_P(nn.Module):
    def __init__(self, 
                num_letters=21,
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                vocab=21,
                k_neighbors=32,
                augment_eps=0.1,
                dropout=0.1):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(
            node_features, 
            edge_features, 
            top_k=4, 
            augment_eps=augment_eps,
            CA_only = True
        )
        self.W_e = nn.Linear(edge_features, hidden_dim)
        self.W_s = nn.Embedding(vocab, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            EncLayer(
                hidden_dim, 
                hidden_dim*2,
                dropout=dropout
            ) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecLayer(
                hidden_dim,
                hidden_dim*3,
                dropout=dropout
            ) for _ in range(num_decoder_layers)
        ])
 
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_states(self, E, S):
        h_V = torch.zeros((E.shape[0], E.shape[1], self.hidden_dim), device=E.device)
        h_E = self.W_e(E)
        return h_V, h_E

    def init_states_pair(self, E, S, w_s):
        h_V = initialize_h_V(self, E, S, w_s, onehot_dim=21, init_ratio=1.0, error_init_ratio=0.0) 
        h_E = self.W_e(E)
        return h_V, h_E

    def _init_decoder(self, h_V, h_E, E_idx, mask, chain_M, S, W_s):
        device = h_V.device
        h_S = W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)
        
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

        chain_M = chain_M * mask
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device))))
        
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        return h_ES, h_EXV_encoder_fw, mask_bw

    def _decoder_step(self, h_V, layer, h_ES, h_EXV_encoder_fw, mask_bw, mask, E_idx):
        h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
        h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
        return layer(h_V, h_ESV, mask)

class AlignedCrossAttention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=256, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_b, mask, align_pairs_list, tm_scores=None):
        # Aligned-wise cross attention

        batch_size = h_a.size(0)
        updates = torch.zeros_like(h_a) 

        for i in range(batch_size):
            align_pairs = align_pairs_list[i]
            a_indices = [p[0] for p in align_pairs]
            b_indices = [p[1] for p in align_pairs]

            Q = self.query(h_a[i, a_indices, :])  # (num_pairs, hidden)
            K = self.key(h_b[i, b_indices, :])    # (num_pairs, hidden)
            V = self.value(h_b[i, b_indices, :])  # (num_pairs, hidden)
            
            attn = torch.einsum('ph,ph->p', Q, K) * self.scale
            attn = F.softmax(attn, dim=0)
            updates[i, a_indices, :] = V * attn.unsqueeze(-1) * tm_scores[i]

        return h_a + self.dropout(updates), h_b

    """A layer that performs attention only at predefined aligned positions"""
    def __init__(self, hidden_dim, attn_dim=256, dropout=0.1):
        super().__init__()
        # Parameter definitions
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_a, h_b, mask, align_pairs_list, tm_scores=None):
        batch_size = h_a.size(0)
        updates = torch.zeros_like(h_a)  # Initialize update tensor
        # Iterate over each sample in the batch
        for i in range(batch_size):
            align_pairs = align_pairs_list[i]  # Get alignment pairs for the current sample
            
            # Extract aligned indices
            a_indices = [p[0] for p in align_pairs]
            b_indices = [p[1] for p in align_pairs]
            
            # Extract vectors at aligned positions (num_pairs, hidden_dim)
            Q = self.query(h_a[i, a_indices, :])  # (num_pairs, hidden)
            K = self.key(h_b[i, b_indices, :])    # (num_pairs, hidden)
            V = self.value(h_b[i, b_indices, :])  # (num_pairs, hidden)
            
            
            attn = torch.einsum('ph,ph->p', Q, K) * self.scale
                
            attn = F.softmax(attn, dim=0)

            # Add the weighted values back to their corresponding positions
            updates[i, a_indices, :] = V * attn.unsqueeze(-1) * tm_scores[i]


        # Update h_a
        return h_a + self.dropout(updates), h_b  # Only h_a is updated
