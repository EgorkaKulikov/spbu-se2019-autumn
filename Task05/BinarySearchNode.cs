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
    }
}
