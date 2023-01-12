from matplotlib import pyplot as plt
import numpy as np


def draw_result(lst_iter, lst_acc, title):
    plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)
    plt.savefig(title+".png")  # should before show method

    plt.show()

if __name__ == '__main__':
    awl_cl = np.load("loss_awl_Cl_RW.npy")
    awl_pr = np.load("loss_awl_Pr_RW.npy")
    awl_mmd = np.load("loss_awl_2layer_mmd.npy")
    # print(awl_pr.shape)
    # print(awl_pr[:100])
    
    # length = min(len(awl_cl), len(awl_pr))
    plt.figure()
    acc_iter = range(4999)
    
    # plt.plot(acc_iter, awl_mmd[:,0], '-r', label='first layer MMD Loss')
    # plt.plot(acc_iter, awl_mmd[:,1], '-b', label='second layer MMD Loss')
    # plt.xlabel("n iteration")
    # plt.ylabel("weight of loss")
    # plt.legend(loc='upper right')
    # title = "2 MMD Loss"
    # plt.title(title)
    

    # plt.plot(acc_iter, awl_cl[:, 0], '-r', label = 'MMD Loss on ClipArt->RealWorld')
    # plt.plot(acc_iter, awl_cl[:, 1], '-b', label = 'Adversarial Loss on ClipArt->RealWorld')
    # plt.xlabel("n iteration")
    # plt.ylabel("weight of loss")
    # plt.legend(loc='upper right')
    # title = "MMD Loss + Adversarial Loss"
    # plt.title(title)
    

    plt.plot(acc_iter, awl_pr[:, 0], '-r', label = 'MMD Loss on Product->RealWorld')
    plt.plot(acc_iter, awl_pr[:, 1], '-b', label = 'Adversarial Loss on Product->RealWorld')
    plt.xlabel("n iteration")
    plt.ylabel("weight of loss")
    plt.legend(loc='upper right')
    title = "MMD Loss + Adversarial Loss"
    plt.title(title)
    
    # title = "compare_loss"
    plt.savefig(title+".png")  # should before show method

    # plt.show()
