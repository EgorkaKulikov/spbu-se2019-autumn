using System;
using System.Threading;

namespace Task05
{
    public abstract class AbstractTree<K, V> : ITree<K, V>
    {
        public class Node
        {
            public K key;
            public V value;
            public Node left = null;
            public Node right = null;

            public Node(K key, V value)
            {
                this.key = key;
                this.value = value;
            }
        }

        protected void Lock(Object obj)
        {
            if (obj == null)
            {
                return;
            }

            Boolean entered = false;

            while (!entered)
            {
                Monitor.Enter(obj, ref entered);
            }
        }

        protected void Unlock(Object obj)
        {
            Monitor.Exit(obj);
        }

        protected Object rootLock = new Object();
        protected Node root;

        public abstract V Find(K key);
        public abstract V Add(K key, V value);
    }
}
