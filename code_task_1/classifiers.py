from torch import nn
from torch.nn import functional as F
class Simple(nn.Module):
    def __init__(self, cl_weight, num_species, num_genera, num_families, num_orders, num_sub_classes):
        super(Simple, self).__init__()
        self.species = nn.Linear(cl_weight.size(1),num_species)
        self.bn1 = nn.BatchNorm1d(num_species)
        self.genera = nn.Linear(num_species,num_genera)
        self.bn2 = nn.BatchNorm1d(num_genera)
        self.families = nn.Linear(num_genera,num_families)
        self.bn3 = nn.BatchNorm1d(num_families)
        self.orders = nn.Linear(num_families,num_orders)
        self.bn4 = nn.BatchNorm1d(num_orders)
        self.sub_classes = nn.Linear(num_orders,num_sub_classes)

    def forward(self, x): 
        species = self.species(x)
        genus = self.genera(F.relu(self.bn1(species)))
        family = self.families(F.relu(self.bn2(genus)))
        order = self.orders(F.relu(self.bn3(family)))
        sub_class = self.sub_classes(F.relu(self.bn4(order)))
        return species, genus, family, order, sub_class

class Parallel(nn.Module):
    def __init__(self, cl_weight, num_species, num_genera, num_families, num_orders, num_sub_classes):
        super(Parallel, self).__init__()
        self.species = nn.Linear(cl_weight.size(1),num_species)
        self.genera = nn.Linear(cl_weight.size(1),num_genera)
        self.families = nn.Linear(cl_weight.size(1),num_families)
        self.orders = nn.Linear(cl_weight.size(1),num_orders)
        self.sub_classes = nn.Linear(cl_weight.size(1),num_sub_classes)

    def forward(self, x):
        species = self.species(x)
        genus = self.genera(x)
        family = self.families(x)
        order = self.orders(x)
        sub_class = self.sub_classes(x)
        return species, genus, family, order, sub_class