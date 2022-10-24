import numpy as np
class Generic() : 
    def __init__(self) :
        self.nb_params=None # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return None
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=None
        return grad_local,grad_entree

class Arctan():
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return np.arctan(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = 1 / (1 + self.save_X**2) * grad_sortie
        return grad_local,grad_entree

class Tanh():
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return np.tanh(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = 1 - np.tanh(self.save_X)**2 * grad_sortie
        return grad_local,grad_entree

class Sigmoid():
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return 1/(1+np.exp(-X))
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = np.exp(-self.save_X) / (1 + np.exp(-self.save_X))**2 * grad_sortie
        return grad_local,grad_entree

class Softmax():
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return np.exp(X) / np.sum(np.exp(X))
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = np.exp(self.save_X) / np.sum(np.exp(self.save_X)) * (grad_sortie - np.sum(np.exp(self.save_X) * grad_sortie) / np.sum(np.exp(self.save_X)))
        return grad_local,grad_entree

class ReLU():
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return np.maximum(X,0)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = (self.save_X > 0) * grad_sortie
        return grad_local,grad_entree

class Dense():
    def __init__(self, nb_entree, nb_sortie) :
        self.nb_params=nb_entree*nb_sortie + nb_sortie # Nombre de parametres de la couche
        self.n_entree=nb_entree
        self.n_sortie=nb_sortie
        self.save_X=None
        self.A=np.random.randn(nb_sortie, nb_entree)
        self.b=np.random.randn(nb_sortie)
    def set_params(self,params) :
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        self.A=params[:self.n_sortie*self.n_entree].reshape(self.n_sortie,self.n_entree)
        self.b=params[self.n_sortie*self.n_entree:]
    def get_params(self) :
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.concatenate((self.A.flatten(),self.b))
    def forward(self,X) :
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return np.dot(self.A,X) + np.outer(self.b,np.ones(X.shape[1]))
    def backward(self,grad_sortie) :
        # retropropagation du gradient sur la couche,
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche
        grad_local=np.dot(grad_sortie,self.save_X.T).flatten()
        grad_local=np.concatenate((grad_local,np.sum(grad_sortie,axis=1)))
        grad_entree=np.dot(self.A.T,grad_sortie)
        return grad_local,grad_entree

class Loss_L2():
    def __init__(self, D) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.D=D
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees, Y est le vecteur des donnees de sortie
        self.save_X=np.copy(X)
        return np.sum((X-self.D)**2)/2
    def backward(self, grad_sortie):  
        # retropropagation du gradient sur la couche, 
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        # grad_entree = phi'(x) * grad_sortie
        grad_entree = self.save_X - self.D
        return grad_local,grad_entree

class Network():
    def __init__(self, list_layers) :
        self.list_layers=list_layers
        self.nb_params=0
        for layer in self.list_layers :
            self.nb_params+=layer.nb_params
    def set_params(self,params) :
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        start=0
        for layer in self.list_layers :
            layer.set_params(params[start:start+layer.nb_params])
            start+=layer.nb_params
    def get_params(self) :
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        params=np.zeros(self.nb_params)
        start=0
        for layer in self.list_layers :
            params[start:start+layer.nb_params]=layer.get_params()
            start+=layer.nb_params
        return params
    def forward(self,X) :
        # calcul du forward, X est le vecteur des donnees d'entrees
        Z=X
        for layer in self.list_layers :
            Z=layer.forward(Z)
        return Z
    def backward(self,grad_sortie) :
        # retropropagation du gradient sur la couche,
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche
        grad_local=np.zeros(self.nb_params)
        start=self.nb_params
        for layer in reversed(self.list_layers) :
            grad_local[start-layer.nb_params:start],grad_sortie=layer.backward(grad_sortie)
            start-=layer.nb_params
        return grad_local,grad_sortie
    def train(self, X, D, nb_epochs, lr) :
        # X est la matrice des donnees d'entree, D est la matrice des donnees de sortie
        # nb_epochs est le nombre d'epochs d'apprentissage
        # lr est le learning rate
        for epoch in range(nb_epochs) :
            Y=self.forward(X)
            loss=np.sum((Y-D)**2)/2
            grad_local,grad_entree=self.backward(Y-D)
            self.set_params(self.get_params()-lr*grad_local)
            print("Epoch ",epoch," loss : ",loss)