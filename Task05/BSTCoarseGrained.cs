using System.Collections.Generic;
using System.Threading;

namespace Task05
{
    public class BSTCoarseGrained<T> : BST<T>
    {
        private Mutex coarseMutex = new Mutex(false);

        public BSTCoarseGrained()
        {
        }
        public BSTCoarseGrained(IEnumerable<(int, T)> elements)
        {
            foreach (var pair in elements)
            {
                Insert(pair.Item1, pair.Item2);
            }
        }

        public override void Insert(int key, T value)
        {
            coarseMutex.WaitOne();
            try
            {
                UpdateTree(null, root, key, value, true);
            }
            finally
            {
                coarseMutex.ReleaseMutex();
            }
        }
        
        public override T Search(int key)
        {
            coarseMutex.WaitOne();
            T result = default(T);
            try
            {
                result = base.Search(key);
            }
            finally
            {
                coarseMutex.ReleaseMutex();
            }
            return result;
        }

        public override void Delete(int key)
        {
            coarseMutex.WaitOne();
            try
            {
                base.Delete(key);
            }
            finally
            {
                coarseMutex.ReleaseMutex();
            }
        }
    }
}