using System;
using System.Collections.Generic;

namespace ParallelTree
{
    class BadParallelTree<T> : Tree<T> where T : IComparable
    //Dat is really bad guy. Dah!
    {
        private readonly object block = new object();

        public override void Add(T value)
        {
            lock (block)
                base.Add(value);
        }

        public override bool Contains(T value)
        {
            bool result;

            lock (block)
                result = base.Contains(value);

            return result;
        }

        public override void Clear()
        {
            lock (block)
                base.Clear();
        }

        protected override List<T> BuildTList()
        {
            List<T> result;

            lock (block)
                result = base.BuildTList();

            return result;
        }
    }
}
