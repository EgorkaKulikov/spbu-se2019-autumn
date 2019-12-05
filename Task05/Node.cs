﻿using System;
 using System.Threading;

 namespace Task05
{
    public class Node<TKey, TValue>
        where TKey : IComparable
        where TValue : struct
    {
        public Node(TKey key, TValue value, Node<TKey, TValue> parent)
        {
            Key = key;
            Value = value;
            Parent = parent;
        }

        public Node<TKey, TValue> Left;
        public Node<TKey, TValue> Right;
        public Node<TKey, TValue> Parent;
        public readonly Mutex NodeMtx = new Mutex();
        public TKey Key;
        public TValue Value;
    }
}