using System;

namespace Task05
{
    public class BinarySearchNode<K, V> : BinaryNode<K, V>
        where K : IComparable<K>
        where V : struct
    {
        public BinarySearchNode(K key, V value)
        {
            Key = key;
            Value = value;
        }

        public void CreateSon(K key, V value, Location locationSon)
        {
            if (locationSon == Location.LeftSubtree)
                SetSon(new BinarySearchNode<K, V>(key, value), SonType.LeftSon);
            else if (locationSon == Location.RightSubtree)
                SetSon(new BinarySearchNode<K, V>(key, value), SonType.RightSon);
        }
    }
}
