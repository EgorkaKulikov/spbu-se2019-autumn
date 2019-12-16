﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Task05
{
    public interface IBinarySearchTree<TKey, TValue>
        where TKey : IComparable
        where TValue : struct
    {
        bool IsEmpty();

        void Insert(TKey key, TValue value);

        TValue? Find(TKey key);

        void Remove(TKey key);
    }
}
