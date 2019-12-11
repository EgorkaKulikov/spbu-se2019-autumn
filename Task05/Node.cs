using System;

namespace Task05
{
    abstract class Node<K, V, NT>
        where K : IComparable<K>
        where NT : Node<K, V, NT>
    {
        public NT left = null;
        public NT right = null;
        public NT parent = null;

        public SonType type {
            get {
                if (parent == null)
                    return SonType.Root;
                if (parent.left == this)
                    return SonType.LeftSon;
                return SonType.RightSon;
            }
        }
        public NT brother {
            get {
                if (type == SonType.LeftSon)
                    return parent.right;
                if (type == SonType.RightSon)
                    return parent.left;
                return null;
            }
        }

        public K key;
        public V value;

        protected abstract NT createNode(K key, V value);

        public void createSon(K key, V value, SonType typeNewSon) {
            setSon(createNode(key, value), typeNewSon);
        }

        public void setSon(NT newSon, SonType typeNewSon)
        {
            if (typeNewSon == SonType.LeftSon) {
                if (left != null)
                    left.parent = null;
                left = newSon;
            }
            else if (typeNewSon == SonType.RightSon) {
                if (right != null)
                    right.parent = null;
                right = newSon;
            }
            else {
                throw new Exception("Function setSon don't expect that" +
                        "typeNewSon can be SonType.Root");
            }
            if (newSon != null)
            {
                if (newSon.type == SonType.LeftSon)
                    newSon.parent.left = null;
                else if (newSon.type == SonType.RightSon)
                    newSon.parent.right = null;
                newSon.parent = this as NT;
            }
        }

        public void setFather(NT newFather, SonType typeThisNode)
        {
            if (newFather != null)
            {
                if (typeThisNode == SonType.LeftSon) {
                    if (newFather.left != null)
                        newFather.left.parent = null;
                    newFather.left = this as NT;
                }
                else if (typeThisNode == SonType.RightSon) {
                    if (newFather.right != null)
                        newFather.right.parent = null;
                    newFather.right = this as NT;
                }
                else {
                    throw new Exception("Function setFather don't expect that" +
                            "typeThisNode can be SonType.Root");
                }
            }
            if (type == SonType.LeftSon)
                parent.left = null;
            else if (type == SonType.RightSon)
                parent.right = null;
            parent = newFather;
        }

        public void moveOn(NT newPlace)
        {
            setFather(newPlace.parent, newPlace.type);
            setSon(newPlace.left, SonType.LeftSon);
            setSon(newPlace.right, SonType.RightSon);
        }

        public SonType findKeyType(K key) {
            if (this.key.CompareTo(key) == 0)
                return SonType.Root;
            else if (this.key.CompareTo(key) > 0)
                return SonType.LeftSon;
            else
                return SonType.RightSon;
        }

        public NT nextNode(K key) {
            if (findKeyType(key) == SonType.Root)
                return null;
            else if (findKeyType(key) == SonType.LeftSon)
                return left;
            else
                return right;
        }
    }
}
