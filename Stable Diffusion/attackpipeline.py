import dataset
import utils
from classifier import *
from train import *
from torch.utils.data import DataLoader , TensorDataset
from train import *
from dataset import *
import numpy as np
import torch.nn as nn
from art.estimators.classification import PyTorchClassifier
from torch import optim



from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (projected_gradient_descent,)


from torchattacks import AutoAttack, EOTPGD

from art.attacks.evasion import FastGradientMethod, DeepFool, ProjectedGradientDescent, SquareAttack, ElasticNet, SignOPTAttack
from art.estimators.classification import PyTorchClassifier



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getConfigs(datasetName, classifierName, attackName=None):
    config = utils.loadconfigYaml("config.yaml")
    
    configAttack = config['attacks'][attackName]
    configAttackGeneral = config['attacks']['general']
    configData = config['dataset'][datasetName]
    configModel = config['classifier'][classifierName]

    return configAttack, configAttackGeneral, configData, configModel

def createClassifier(submodel, configsmodel):

    r""" Create a pytorch classifier to use ART attacks

    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(submodel.parameters(), lr=configsmodel['learning_rate'], momentum=configsmodel['momentum'])

    classifier = PyTorchClassifier(model=submodel,clip_values=(0, 1),loss=criterion,optimizer=optimizer,input_shape=(1, 28,28),nb_classes=10,)

    return classifier

def createAdversarialData(attack, testset, batch_size):
         
    for i, (images, label) in enumerate(testset):
        if i == 0:
            x_test_adv = torch.tensor(attack.generate(x = images.cpu().detach().numpy()))
            l = label
        else:
            x_test_adv = torch.cat ([x_test_adv,torch.tensor(attack.generate(x = images.cpu().detach().numpy()))])
            l = torch.cat ([l,label ])



    return x_test_adv , l

def FGSM_Attack_CH(submodel, datasetname, classifiername, testset, batchSize, type=None):
    """
    FGSM ATTACK
    """
    
    print("Attack using FGSM CH....")

    for i, (images, label) in enumerate(testset):
        
        images, label = images.to(device), label.to(device)

        if i == 0:
            advtestset = fast_gradient_method(model_fn=submodel, x=images, norm=np.inf,eps=0.3).detach().clone()
            l = label
        else:
            advtestset = torch.cat ([advtestset,fast_gradient_method(model_fn=submodel, x=images, norm=np.inf,eps=0.3).detach().clone()])
            l = torch.cat ([l,label ])
    

    
    print("Attack conclude....")
    
    return advtestset , l
    
def PGD_Attack_CH(submodel, datasetname, classifiername, testset, batchSize, type):
    
    
    print("Attack using PGD CH....")
    #get configs


    for i, (images, label) in enumerate(testset):
        
        images, label = images.to(device), label.to(device)

        if i == 0:
            advtestset = projected_gradient_descent(model_fn=submodel,x=images,eps=0.3,norm=np.inf, eps_iter=0.05, nb_iter = 5).detach().clone()
            l = label
        else:
            advtestset = torch.cat ([advtestset,projected_gradient_descent(model_fn=submodel,x=images,eps=0.3,norm=np.inf, eps_iter=0.05, nb_iter = 5).detach().clone()])
            l = torch.cat ([l,label])
    
   
    print("Attack conclude....")
    
    return advtestset , l

def FGSM_Attack(submodel, datasetname, classifiername, testset, batchSize, type):
    
    print("Attack using FGSM....")
    #get configs
    configAttack, _, _, configModel = getConfigs(datasetname, type, "fgsm")

    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = FastGradientMethod(estimator=classifier, eps=configAttack['eps'], norm=configAttack['norm'])

    # Attaching testset and create a dataloader with adv_images
    advtestloader, l = createAdversarialData(attack, testset, batchSize)
    
    print("Attack conclude....")
    
    return advtestloader, l


def PGD_Attack(submodel, datasetname, classifiername,testset, batchSize, type):    

    print("Attack using PGD....")
    #get configs
    _, _, _, configModel = getConfigs(datasetname, type, 'cw')

    # Defining the model to attack
    classifier = createClassifier(submodel, configModel)
    
    #Defining the attack
    attack = ProjectedGradientDescent(classifier,norm=np.inf,eps=0.3,eps_step=0.01,max_iter=20,targeted=False,num_random_init=5)

    # Attaching testset and create a dataloader with adv_images
    advtestloader,l = createAdversarialData(attack, testset, batchSize)

    print("Attack conclude....")

    return advtestloader, l