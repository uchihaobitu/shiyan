# CR-VAE

Codes of CR-VAE. Run CRVAE_demo.py to obtain some demo results on Henon data.

修改了Model/cgru文件里 引入了torchlinear的三个函数
均是torch格式的
使用新增的train_phase3计算

在新增的train_phase4中使用你之前的

    def cal_loss_batched(Xs, connections):
    
        batch_size, n, d = Xs.shape
        Id = np.eye(d).astype(np.float64)
        Ids = np.broadcast_to(Id, (batch_size, d, d))
        covs = np.einsum('bni,bnj->bij', Xs, Xs) / n
        difs = Ids - connections
        rhs = np.einsum('bij,bjk->bik', covs, difs)
        losses = 0.5 * np.einsum('bij,bji->b', difs, rhs) / (d * d)
        G_losses = -rhs

        return losses, G_losses

我把他改成了torch版本的

    def cal_loss_batched_torch(Xs, connections,device='cuda:1'):
        batch_size, n, d = Xs.shape
        Id = torch.eye(d, dtype=torch.float32).to(device)
        Ids = Id.expand(batch_size, d, d)  # 使用expand来广播到每个批次
    
        covs = torch.einsum('bni,bnj->bij', Xs, Xs) / n
        difs = Ids - connections
        rhs = torch.einsum('bij,bjk->bik', covs, difs)
        losses = 0.5 * torch.einsum('bij,bji->b', difs, rhs) / (d * d)
        G_losses = -rhs
    
        return losses, G_losses

阶段一 问题一遗留的：
    
    用我给的torchlinear进行修改 既然没有groundtruth
    只看跑出来的图的效果，用他源码（最开始的源码）不进行任何改变跑一个
    GC_yuan.npy用改进后的跑一个的GC_my.npy 然后绘图对比
    这次只改了一小部分 因果矩阵全为1不合理
    
阶段一 问题二 
    
    暂时先不管我刚刚说的那个新加的方法，主要改动train_phase2和新的vr4那个
    让其输入的X不变因为里面有他自己的error项pred变化+X
    同样传入参数X，新加一个传入项train_phase1产生的因果图
    让其对因果图微调
    输出原函数会输出的产生的时序+新的因果图npy
    注意区分训练集和验证集
    原代码只给了train函数没给test函数 写一个
    两个模型里源码都有test阶段可以参考
    原train_phase会产生时序源代码也有评价指标
    用这个当做指标

争取周四给一个大概的结果要汇报
    
    

    
