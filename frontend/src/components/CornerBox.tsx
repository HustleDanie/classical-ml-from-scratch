import { ReactNode } from 'react';

interface CornerBoxProps {
  children?: ReactNode;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  always?: boolean;
  color?: string;
}

const sizeMap = {
  sm: 'w-3 h-3',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
};

export function CornerBox({
  children,
  className = '',
  size = 'md',
  always = true,
  color,
}: CornerBoxProps) {
  const dim = sizeMap[size];
  const baseColor = color
    ? color
    : always
      ? 'border-black dark:border-white'
      : 'border-transparent group-hover:border-black dark:group-hover:border-white';
  const transition = always ? '' : 'transition-colors duration-300';

  return (
    <div className={`relative ${className}`}>
      <div className={`absolute -top-1 -left-1 ${dim} border-t-2 border-l-2 ${baseColor} ${transition}`} />
      <div className={`absolute -top-1 -right-1 ${dim} border-t-2 border-r-2 ${baseColor} ${transition}`} />
      <div className={`absolute -bottom-1 -left-1 ${dim} border-b-2 border-l-2 ${baseColor} ${transition}`} />
      <div className={`absolute -bottom-1 -right-1 ${dim} border-b-2 border-r-2 ${baseColor} ${transition}`} />
      {children}
    </div>
  );
}
