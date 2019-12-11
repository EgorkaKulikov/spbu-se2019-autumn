using System;
using System.Threading;

namespace Task05
{
    class BinarySearchNode<K, V> : Node<K, V, BinarySearchNode<K, V>>
        where K : IComparable<K>
    {
        public Mutex mutex = new Mutex();

        public BinarySearchNode(K key, V value)
        {
            this.key = key;
            this.value = value;
        }

        protected override BinarySearchNode<K, V> createNode(K key, V value) => new BinarySearchNode<K, V>(key, value);

        public override BinarySearchNode<K, V> copy() => new BinarySearchNode<K, V>(key, value);

        public override bool Equals(object other) =>
            (other is BinarySearchNode<K, V> &&
                    (other as BinarySearchNode<K, V>).left == left &&
                    (other as BinarySearchNode<K, V>).right == right &&
                    (other as BinarySearchNode<K, V>).key.CompareTo(key) == 0 &&
                    (other as BinarySearchNode<K, V>).value.Equals(value));
    }
}
