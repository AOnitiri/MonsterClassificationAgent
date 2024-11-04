import datetime


class MonsterClassificationAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, samples, new_monster):
        #Add your code here!
        #
        #The first parameter to this method will be a labeled list of samples in the form of
        #a list of 2-tuples. The first item in each 2-tuple will be a dictionary representing
        #the parameters of a particular monster. The second item in each 2-tuple will be a
        #boolean indicating whether this is an example of this species or not.
        #
        #The second parameter will be a dictionary representing a newly observed monster.
        #
        #Your function should return True or False as a guess as to whether or not this new
        #monster is an instance of the same species as that represented by the list.
        #
        start_time = datetime.datetime.now()
        print(start_time)


        tree = self.classification_tree(samples, list(new_monster.keys()))
        print("tree: ", tree)
        result =  self.classify(new_monster, tree)
        end_time = datetime.datetime.now() - start_time

        print(end_time)
        return result
    
    def classification_tree(self, samples, attributes):
        labels = [label for _, label in samples]

        if all(labels):
            return True 
        if not any(labels):
            return False 
        if not attributes:
            return max(set(labels), key=labels.count)
        
        monster_attributes = self.select_monster_attribute(samples, attributes)
        tree = {monster_attributes: {}}
        
        values = set(monster[monster_attributes] for monster, _ in samples)

        for value in values:

            subset = [(monster, label) for monster, label in samples if monster[monster_attributes] == value]
            if not subset:
                
                majority_class = max(set(labels), key=labels.count)
                tree[monster_attributes][value] = majority_class
            else:
                
                remaining_attributes = attributes.copy()
                remaining_attributes.remove(monster_attributes)
                subtree = self.classification_tree(subset, remaining_attributes)
                tree[monster_attributes][value] = subtree

        return tree

    def select_monster_attribute(self, samples, attributes):
        base_entropy = self.entropy(samples)
        best_info_gain = -1

        monster_attributes = None
        for attr in attributes:
            attr_entropy = 0
            values = set(monster[attr] for monster, _ in samples)
            for value in values:
                subset = [(monster, label) for monster, label in samples if monster[attr] == value]
                prob = len(subset) / len(samples)
                attr_entropy += prob * self.entropy(subset)

            info_gain = base_entropy - attr_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                monster_attributes = attr
        return monster_attributes

    def entropy(self, samples):
        from math import log2
        num_of_samples = len(samples)
        if num_of_samples == 0:
            return 0
        positives = sum(1 for _, label in samples if label)
        negatives = num_of_samples - positives

        entropy = 0
        for count in [positives, negatives]:
            if count == 0:
                continue
            prob = count / num_of_samples
            entropy -= prob * log2(prob)
        return entropy

    def classify(self, instance, tree):
        if tree == True or tree == False:
            return tree
        if not isinstance(tree, dict):
            return tree
        
        attr = next(iter(tree))
        
        value = instance.get(attr)
        print(attr, value)
        subtree = tree[attr].get(value)
        print("subtree", subtree)
        if subtree is None:
            return False
        else:
            return self.classify(instance, subtree)