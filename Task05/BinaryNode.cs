using System;

namespace Task05
{
    abstract public class BinaryNode<K, V>
        where K : IComparable<K>
        where V : struct
    {
        public BinaryNode<K, V> Left = null;
        public BinaryNode<K, V> Right = null;
        public BinaryNode<K, V> Parent = null;
        public K Key;
        public V Value;

        public SonType Type {
            get {
                if (Parent == null)
                    return SonType.Root;
                if (Parent.Left == this)
                    return SonType.LeftSon;
                return SonType.RightSon;
            }
        }

        public void SetSon(BinaryNode<K, V> newSon, SonType typeNewSon)
        {
            if (typeNewSon == SonType.LeftSon) {
                if (Left != null)
                    Left.Parent = null;
                Left = newSon;
            }
            else if (typeNewSon == SonType.RightSon) {
                if (Right != null)
                    Right.Parent = null;
                Right = newSon;
            }
            if (newSon != null)
            {
                if (newSon.Type == SonType.LeftSon)
                    newSon.Parent.Left = null;
                else if (newSon.Type == SonType.RightSon)
                    newSon.Parent.Right = null;
                newSon.Parent = this;
            }
        }

        public void SetFather(BinaryNode<K, V> newFather, SonType typeThisNode)
        {
            if (newFather != null)
            {
                if (typeThisNode == SonType.LeftSon) {
                    if (newFather.Left != null)
                        newFather.Left.Parent = null;
                    newFather.Left = this;
                }
                else if (typeThisNode == SonType.RightSon) {
                    if (newFather.Right != null)
                        newFather.Right.Parent = null;
                    newFather.Right = this;
                }
            }
            if (Type == SonType.LeftSon)
                Parent.Left = null;
            else if (Type == SonType.RightSon)
                Parent.Right = null;
            Parent = newFather;
        }

        public void MoveOn(BinaryNode<K, V> newPlace)
        {
            SetFather(newPlace.Parent, newPlace.Type);
            SetSon(newPlace.Left, SonType.LeftSon);
            SetSon(newPlace.Right, SonType.RightSon);
        }

        public Location FindKey(K key) {
            if (this.Key.CompareTo(key) == 0)
                return Location.ThisNode;
            else if (this.Key.CompareTo(key) > 0)
                return Location.LeftSubtree;
            else
                return Location.RightSubtree;
        }

        public BinaryNode<K, V> GetNextNode(K key) {
            if (FindKey(key) == Location.ThisNode)
                return null;
            else if (FindKey(key) == Location.LeftSubtree)
                return this.Left;
            else
                return this.Right;
        }
    }
}
