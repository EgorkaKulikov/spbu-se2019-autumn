using System;
using System.Threading;

namespace Task05
{
    public class LockedBinarySearchNode<K, V> : BinaryNode<K, V>, ILocked
        where K : IComparable<K>
        where V : struct
    {
        private Mutex Mutex = new Mutex();

        public LockedBinarySearchNode(K key, V value)
        {
            Key = key;
            Value = value;
        }

        public void Lock()
        {
            Mutex.WaitOne();
        }

        public void Unlock()
        {
            Mutex.ReleaseMutex();
        }

        public void CreateSon(K key, V value, Location locationSon)
        {
            if (locationSon == Location.LeftSubtree)
                SetSon(new LockedBinarySearchNode<K, V>(key, value), SonType.LeftSon);
            else if (locationSon == Location.RightSubtree)
                SetSon(new LockedBinarySearchNode<K, V>(key, value), SonType.RightSon);
        }
    }
}
